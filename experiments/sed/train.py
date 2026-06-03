"""sed/train.py — single trainer driven by SedConfig.

Replaces the train_epoch / main blocks duplicated in 8 SED scripts.
Public API: train_one_fold(cfg, fold_id) → ckpt path

Reuses our existing exp169v2 anchor cache (random anchors per file). Future
work: pluggable cache backend so Tucker-style on-the-fly random crop is
supported.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from .config import SedConfig, exp175_tucker_actually
from .data import (
    DATA,
    ROOT,
    CachedDataset,
    PseudoDataset,
    TwoStreamBatchSampler,
    build_label_lookups,
    parse_secondary,
    split_ss_legacy,
    split_ss_per_fold,
)
from .model import DistilledSED, hybrid_loss

CACHE_DIR_DEFAULT = ROOT / "experiments" / "_data_pipelines" / "exp169v2_outputs"
BG_PATH = ROOT / "experiments" / "_data_pipelines" / "exp49_outputs" / "bg_quiet_2025.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
SEED_DEFAULT = 42
SOURCE_DIR = DATA / "train_soundscapes"


def _load_anchor_cache(cfg=None):
    cache_dir = CACHE_DIR_DEFAULT
    _cd = getattr(cfg, "ANCHOR_CACHE_DIR", None)
    if _cd:
        cache_dir = _cd if isinstance(_cd, Path) else Path(_cd)
        if not cache_dir.is_absolute():
            cache_dir = ROOT / cache_dir
    print(f"[cache] loading anchors from {cache_dir}", flush=True)
    cache = np.load(cache_dir / "anchors.npz", allow_pickle=True)
    return dict(
        files=cache["files"],
        is_ss=cache["is_ss"],
        ss_endsec=cache["ss_endsec"],
        primary_idx=cache["primary_idx"],
        ss_label_str=cache["ss_label_str"],
        n_anchors=cache["n_anchors"],
        anchor_off=cache["anchor_off"],
        anchor_emb=cache["anchor_emb"],
    )


def _build_secondary_map(files, is_ss, l2i):
    ta_df = pd.read_csv(DATA / "train.csv")
    ta_df = ta_df[ta_df["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    ta_df["secondary_list"] = ta_df["secondary_labels"].apply(parse_secondary)
    fname_to_secondary = dict(zip(ta_df["filename"], ta_df["secondary_list"]))
    return {
        int(gi): fname_to_secondary.get(files[gi], [])
        for gi in np.where(is_ss == 0)[0]
    }


def _build_ss_labels_map(files, is_ss, ss_label_str, l2i):
    out: dict[int, list[str]] = {}
    for gi in np.where(is_ss == 1)[0]:
        s = ss_label_str[gi]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            out[int(gi)] = []
        else:
            out[int(gi)] = [t for t in str(s).split(";") if t.strip()]
    return out


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate(model, ds, dev, batch=32, max_n=2000):
    from sklearn.metrics import roc_auc_score
    model.eval()
    n = min(max_n, len(ds))
    rng = np.random.RandomState(123)
    pick = rng.choice(len(ds), n, replace=False) if len(ds) > n else np.arange(len(ds))
    n_cls = ds.n_cls
    preds = np.zeros((n, n_cls), dtype=np.float32)
    Y = np.zeros((n, n_cls), dtype=np.uint8)
    wavs_buf, ys_buf, idx_buf = [], [], []
    for i, k in enumerate(pick):
        wav, _, y, _, _ = ds[int(k)]
        wavs_buf.append(wav)
        ys_buf.append(y)
        idx_buf.append(i)
        if len(wavs_buf) >= batch or i + 1 == n:
            x = torch.stack(wavs_buf).to(dev)
            clip_l, _, _ = model(x)
            p = torch.sigmoid(clip_l).cpu().numpy()
            for j, ri in enumerate(idx_buf):
                preds[ri] = p[j]
                Y[ri] = ys_buf[j].numpy()
            wavs_buf, ys_buf, idx_buf = [], [], []
    aucs = []
    for c in range(n_cls):
        s = Y[:, c].sum()
        if s == 0 or s == len(Y):
            continue
        try:
            aucs.append(roc_auc_score(Y[:, c], preds[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


def _maybe_remap_legacy_bottleneck(state_dict: dict, model: DistilledSED) -> dict:
    """exp169v2 had bottleneck.0=Linear; new cfg has bottleneck.1=Linear (Tucker spec).
    Auto-remap when fine-tuning from legacy ckpts.
    """
    model_keys = set(model.state_dict().keys())
    if "bottleneck.1.weight" in model_keys and "bottleneck.0.weight" not in state_dict:
        return state_dict
    if "bottleneck.0.weight" in state_dict and "bottleneck.1.weight" in model_keys:
        out = {}
        for k, v in state_dict.items():
            if k.startswith("bottleneck.0."):
                out[k.replace("bottleneck.0.", "bottleneck.1.")] = v
            else:
                out[k] = v
        return out
    return state_dict


def _train_one_epoch(model, loader, opt, sched, scaler, dev, cfg, pos_weight=None):
    model.train()
    tot, n, nan_skip = 0.0, 0, 0
    bce_c_sum, bce_f_sum, mse_sum = 0.0, 0.0, 0.0
    for wav, perch, y, _, source_kind in loader:
        wav = wav.to(dev, non_blocking=True)
        perch = perch.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        is_pseudo = (source_kind == 2).to(dev, non_blocking=True)
        if not torch.isfinite(wav).all():
            wav = torch.nan_to_num(wav, 0.0, 1.0, -1.0)
        opt.zero_grad(set_to_none=True)
        if cfg.USE_AMP:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                clip_l, frame_l, demb = model(wav, training_aug=True)
                loss, bc, bf, ms = hybrid_loss(clip_l, frame_l, demb, perch, y, cfg, is_pseudo, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                nan_skip += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            if not torch.isfinite(gn):
                nan_skip += 1
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
        else:
            clip_l, frame_l, demb = model(wav, training_aug=True)
            loss, bc, bf, ms = hybrid_loss(clip_l, frame_l, demb, perch, y, cfg, is_pseudo, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                nan_skip += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item() * wav.size(0)
        n += wav.size(0)
        bce_c_sum += bc
        bce_f_sum += bf
        mse_sum += ms
    nb = max(1, len(loader))
    return tot / max(n, 1), bce_c_sum / nb, bce_f_sum / nb, mse_sum / nb, nan_skip


def train_one_fold(cfg: SedConfig, fold_id: int):
    """Run one fold end-to-end. Returns dict with {history, best, ckpt_path}."""
    seed = cfg.seed + fold_id
    _set_seed(seed)
    primary_labels, l2i = build_label_lookups()
    n_cls = len(l2i)

    cache = _load_anchor_cache(cfg)
    files = cache["files"]
    is_ss = cache["is_ss"]
    primary_idx = cache["primary_idx"]
    ss_label_str = cache["ss_label_str"]
    n_anchors = cache["n_anchors"]
    anchor_off = cache["anchor_off"]
    anchor_emb = cache["anchor_emb"]

    secondary_map = _build_secondary_map(files, is_ss, l2i)
    ss_labels_map = _build_ss_labels_map(files, is_ss, ss_label_str, l2i)

    # Fold split: focal stratified by primary, SS legacy or per-fold
    ta_indices = np.where(is_ss == 0)[0]
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.seed)
    folds = list(skf.split(ta_indices, primary_idx[ta_indices]))
    tr_local, va_local = folds[fold_id]
    tr_global = ta_indices[tr_local]
    va_global = ta_indices[va_local]

    if cfg.EVAL_SS_N_FILES is not None:
        ss_train_indices, ss_eval_indices = split_ss_legacy(
            is_ss, files, cfg.EVAL_SS_N_FILES, cfg.seed
        )
    else:
        ss_folds = split_ss_per_fold(is_ss, files, cfg.N_FOLDS, cfg.seed)
        ss_train_indices, ss_eval_indices = ss_folds[fold_id]

    train_idx = np.concatenate([tr_global, ss_train_indices])
    print(
        f"[fold {fold_id}] train_idx {len(train_idx)} (TA {len(tr_global)} + SS {len(ss_train_indices)})",
        flush=True,
    )

    # MIN_SAMPLE rare upsampling
    if cfg.MIN_SAMPLE > 0:
        ta_in_train = train_idx[: len(tr_global)]
        primary_at_ta = primary_idx[ta_in_train]
        cls_count = np.bincount(primary_at_ta, minlength=n_cls)
        extra = []
        for c in range(n_cls):
            n_have = int(cls_count[c])
            if 0 < n_have < cfg.MIN_SAMPLE:
                rows = ta_in_train[primary_at_ta == c]
                extra.append(np.random.choice(rows, size=cfg.MIN_SAMPLE - n_have, replace=True))
        if extra:
            train_idx = np.concatenate([train_idx] + extra)
            print(f"  MIN_SAMPLE={cfg.MIN_SAMPLE} upsample: +{sum(len(e) for e in extra)} rows", flush=True)

    bg_pool = np.load(BG_PATH)["windows"] if BG_PATH.exists() else None

    ds_train = CachedDataset(
        cfg, train_idx, l2i, True,
        secondary_map, ss_labels_map, primary_idx, is_ss, files,
        n_anchors, anchor_off, anchor_emb, bg_pool,
    )
    ds_val_ta = CachedDataset(
        cfg, va_global, l2i, False,
        secondary_map, ss_labels_map, primary_idx, is_ss, files,
        n_anchors, anchor_off, anchor_emb, None,
    )
    ds_val_ss = CachedDataset(
        cfg, ss_eval_indices, l2i, False,
        secondary_map, ss_labels_map, primary_idx, is_ss, files,
        n_anchors, anchor_off, anchor_emb, None,
    )

    # Sampler / pseudo dataset
    if cfg.pseudo and cfg.pseudo.pseudo_npz:
        p = np.load(ROOT / cfg.pseudo.pseudo_npz, allow_pickle=True)
        ds_pseudo = PseudoDataset(
            cfg, p["files"], p["kept_file_idx"], p["kept_end_secs"],
            p["kept_probs"], SOURCE_DIR, train=True, n_cls=n_cls,
        )
        ds_combined = ConcatDataset([ds_train, ds_pseudo])
        nbatches = max(100, len(tr_global) // cfg.BATCH)
        # Noisy Student weighted sampler (Boredom 2025 1st recipe)
        pseudo_weights = None
        if getattr(cfg, "use_pseudo_weighted_sampler", False):
            if "confidence_weights" in p.files:
                pseudo_weights = p["confidence_weights"]
                print(f"[fold {fold_id}] using pseudo_weighted_sampler "
                      f"(weights mean={pseudo_weights.mean():.3f}, "
                      f"max={pseudo_weights.max():.3f})", flush=True)
            else:
                # fallback: derive from sum of kept_probs
                pseudo_weights = p["kept_probs"].sum(axis=1).astype(np.float32)
                print(f"[fold {fold_id}] pseudo_weights derived from kept_probs.sum",
                      flush=True)
        bs = TwoStreamBatchSampler(
            n_cached=len(ds_train), n_pseudo=len(ds_pseudo),
            batch_size=cfg.BATCH, num_batches=nbatches,
            pseudo_share=cfg.pseudo.pseudo_share_per_batch, seed=cfg.seed + fold_id,
            pseudo_weights=pseudo_weights,
        )
        loader = DataLoader(ds_combined, batch_sampler=bs, num_workers=NUM_WORKERS,
                            pin_memory=True, persistent_workers=True)
    else:
        weights = np.ones(len(train_idx), dtype=np.float64)
        n_ta_now = (is_ss[train_idx] == 0).sum()
        n_ss_now = (is_ss[train_idx] == 1).sum()
        ss_w = (n_ta_now / max(1, n_ss_now)) * (cfg.SOURCE_SHARE_SC / cfg.SOURCE_SHARE_FOCAL)
        weights[is_ss[train_idx] == 1] = ss_w
        sampler = WeightedRandomSampler(weights, num_samples=len(tr_global), replacement=True)
        loader = DataLoader(ds_train, batch_size=cfg.BATCH, sampler=sampler,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                            persistent_workers=True)

    # Model
    model = DistilledSED(cfg, n_cls=n_cls).to(DEVICE)

    # Fine-tune: load base ckpt
    if cfg.finetune and cfg.finetune.base_ckpt_dir:
        base_ckpt = ROOT / cfg.finetune.base_ckpt_dir / f"fold{fold_id}" / "best_ckpt.pt"
        if base_ckpt.exists():
            sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)
            raw_sd = sd["state_dict"]
            if cfg.finetune.bottleneck_remap:
                raw_sd = _maybe_remap_legacy_bottleneck(raw_sd, model)
            missing, unexpected = model.load_state_dict(raw_sd, strict=False)
            print(f"  finetune load: {base_ckpt.name} missing={len(missing)} unexpected={len(unexpected)}",
                  flush=True)

    print(f"  params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    # Optimizer / schedule
    lr = cfg.finetune.finetune_lr if cfg.finetune else cfg.LR
    epochs = cfg.finetune.finetune_epochs if cfg.finetune else cfg.EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WD)
    steps_per_ep = max(1, len(loader))
    warmup_steps = steps_per_ep * cfg.WARMUP_EPOCHS
    total_steps = steps_per_ep * epochs
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=cfg.WARMUP_START_FACTOR, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.MIN_LR
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.USE_AMP else None

    out_dir = cfg.resolved_output_dir() / f"fold{fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class pos_weight for BCE upweighting on rare/ultra-rare positives
    # (Sprint 2026-05-17 v11 direction. F9 unified mechanism fix.)
    pos_weight = None
    if cfg.pos_weight_ultra_rare > 1.0 or cfg.pos_weight_rare > 1.0:
        import pandas as pd
        _train = pd.read_csv(ROOT / "data/birdclef-2026/train.csv")
        _n_train_d = _train.groupby("primary_label").size().to_dict()
        _primary_labels = pd.read_csv(ROOT / "data/birdclef-2026/sample_submission.csv", nrows=1).columns[1:].tolist()
        _n_per_sp = np.array([_n_train_d.get(p, 0) for p in _primary_labels])
        pos_weight = torch.ones(n_cls, device=DEVICE)
        pos_weight[torch.from_numpy(_n_per_sp < 5).to(DEVICE)] = cfg.pos_weight_ultra_rare
        pos_weight[torch.from_numpy((_n_per_sp >= 5) & (_n_per_sp < 20)).to(DEVICE)] = cfg.pos_weight_rare
        n_ur = (_n_per_sp < 5).sum()
        n_r = ((_n_per_sp >= 5) & (_n_per_sp < 20)).sum()
        print(f"  pos_weight: ultra-rare ×{cfg.pos_weight_ultra_rare} ({n_ur} cls), rare ×{cfg.pos_weight_rare} ({n_r} cls)", flush=True)

    history = []
    best = {"val_SS": -1.0, "val_TA": -1.0, "epoch": 0}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, bc, bf, ms, nans = _train_one_epoch(model, loader, opt, sched, scaler, DEVICE, cfg, pos_weight=pos_weight)
        ta_auc, n_ta = _evaluate(model, ds_val_ta, DEVICE, max_n=1500)
        ss_auc, n_ss = _evaluate(model, ds_val_ss, DEVICE, max_n=200)
        dt = time.time() - t0
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"], loss=tr_loss,
                   bce_clip=bc, bce_frame=bf, mse=ms, val_TA=ta_auc, n_TA=n_ta,
                   val_SS=ss_auc, n_SS=n_ss, time_s=dt, nan_skip=nans)
        history.append(row)

        # Checkpoint criterion
        if cfg.CHECKPOINT_CRITERION == "val_ss":
            current = ss_auc
        elif cfg.CHECKPOINT_CRITERION == "val_ta":
            current = ta_auc
        else:  # non_s22_macro_auc — for now use val_SS as proxy
            current = ss_auc
        tag = ""
        if current > best["val_SS"]:
            best = {"val_SS": ss_auc, "val_TA": ta_auc, "epoch": ep}
            tag = " *best"
            torch.save(
                {"state_dict": model.state_dict(), "epoch": ep,
                 "val_TA": ta_auc, "val_SS": ss_auc, "config": cfg.__dict__},
                out_dir / "best_ckpt.pt",
            )
        print(f"  ep{ep:02d}  lr {opt.param_groups[0]['lr']:.5f}  loss {tr_loss:.4f}  "
              f"val_TA {ta_auc:.4f} ({n_ta})  val_SS {ss_auc:.4f} ({n_ss})  nan {nans}  ({dt:.0f}s){tag}",
              flush=True)

    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history, "best": best, "config": cfg.__dict__}, f, indent=2, default=str)
    print(f"FOLD {fold_id} done. best ep {best['epoch']} val_TA {best['val_TA']:.4f} val_SS {best['val_SS']:.4f}",
          flush=True)
    return {"history": history, "best": best, "ckpt": out_dir / "best_ckpt.pt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--config", type=str, default="exp175_tucker_actually",
                        help="config function name in sed.config")
    parser.add_argument("--seed", type=int, default=None,
                        help="override cfg.seed (model init seed; fold split unchanged)")
    args = parser.parse_args()
    from . import config as cfg_mod
    cfg_fn = getattr(cfg_mod, args.config)
    cfg = cfg_fn()
    if args.seed is not None:
        from dataclasses import replace
        cfg = replace(cfg, seed=args.seed)
    folds = list(range(cfg.N_FOLDS)) if args.fold < 0 else [args.fold]
    print(f"config: {args.config}  seed: {cfg.seed}  output: {cfg.resolved_output_dir()}", flush=True)
    for fk in folds:
        train_one_fold(cfg, fk)


if __name__ == "__main__":
    main()

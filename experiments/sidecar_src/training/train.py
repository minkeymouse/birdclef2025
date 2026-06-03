import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from sidecar_src.datasets.audio_dataset import SoundscapeWindowDataset
from sidecar_src.models.convnext_sed import build_model
from sidecar_src.utils.config import ensure_dir, load_config
from sidecar_src.utils.metric import macro_auc
from sidecar_src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def make_loader(ds, cfg, train: bool, weights: np.ndarray | None = None) -> DataLoader:
    key = "train" if train else "valid"
    batch_size = int(cfg[key].get("batch_size", cfg["train"]["batch_size"]))
    num_workers = int(cfg[key].get("num_workers", cfg["train"].get("num_workers", 4)))
    sampler = None
    shuffle = train
    if train and weights is not None:
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


def load_optional_array(path_value: str | None, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float32)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Array shape mismatch for {path}: expected={shape}, actual={arr.shape}")
    return arr


def add_source_column(meta: pd.DataFrame, source: str) -> pd.DataFrame:
    meta = meta.copy()
    if "source" not in meta.columns:
        meta["source"] = source
    return meta


def concat_training_sources(
    sound_meta: pd.DataFrame,
    sound_y: np.ndarray,
    sound_mask: np.ndarray,
    cfg: dict,
    fold: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metas = [add_source_column(sound_meta, "soundscape")]
    targets = [sound_y.astype(np.float32)]
    masks = [sound_mask.astype(np.float32)]

    train_audio_cfg = cfg.get("train_audio", {})
    if not bool(train_audio_cfg.get("enabled", False)):
        return metas[0].reset_index(drop=True), targets[0], masks[0]

    data_cfg = cfg["data"]
    audio_meta_path = data_cfg.get("train_audio_meta")
    audio_y_path = data_cfg.get("train_audio_targets")
    audio_mask_path = data_cfg.get("train_audio_target_masks")
    if not audio_meta_path or not audio_y_path:
        raise FileNotFoundError("train_audio.enabled=True requires data.train_audio_meta and data.train_audio_targets")
    audio_meta = pd.read_parquet(audio_meta_path)
    audio_y = np.load(audio_y_path).astype(np.float32)
    audio_mask = load_optional_array(audio_mask_path, audio_y.shape)
    if audio_mask is None:
        audio_mask = np.ones_like(audio_y, dtype=np.float32)
    if len(audio_meta) != audio_y.shape[0]:
        raise ValueError(f"train_audio rows mismatch: meta={len(audio_meta)}, y={audio_y.shape[0]}")

    if "fold" in audio_meta.columns:
        keep = np.where(audio_meta["fold"].to_numpy() != fold)[0]
        audio_meta = audio_meta.iloc[keep].reset_index(drop=True)
        audio_y = audio_y[keep]
        audio_mask = audio_mask[keep]
    max_rows = int(train_audio_cfg.get("max_rows_per_fold", 0) or 0)
    if max_rows > 0 and len(audio_meta) > max_rows:
        rng = np.random.default_rng(int(cfg.get("seed", 7177)) + 1000 + fold)
        keep = np.sort(rng.choice(len(audio_meta), size=max_rows, replace=False))
        audio_meta = audio_meta.iloc[keep].reset_index(drop=True)
        audio_y = audio_y[keep]
        audio_mask = audio_mask[keep]

    metas.append(add_source_column(audio_meta, "train_audio"))
    targets.append(audio_y)
    masks.append(audio_mask)
    out_meta = pd.concat(metas, axis=0, ignore_index=True)
    out_y = np.concatenate(targets, axis=0).astype(np.float32)
    out_mask = np.concatenate(masks, axis=0).astype(np.float32)
    return out_meta, out_y, out_mask


def make_sample_weights(meta: pd.DataFrame, y: np.ndarray, cfg: dict) -> np.ndarray:
    positive = (y.sum(axis=1) > 0).astype(np.float32)
    weights = 1.0 + positive * float(cfg["train"].get("positive_sample_boost", 0.0))
    source_weights = cfg["train"].get("source_weights", {})
    if source_weights and "source" in meta.columns:
        source_scale = meta["source"].map(lambda s: float(source_weights.get(str(s), 1.0))).to_numpy(np.float32)
        weights = weights * source_scale
    return weights.astype(np.float32)


@torch.no_grad()
def validate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    preds, targets, row_ids = [], [], []
    for batch in tqdm(loader, desc="valid", leave=False):
        x = batch["image"].to(device, non_blocking=True).float()
        logits = model(x)
        preds.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["target"].numpy())
        row_ids.extend([str(x) for x in batch["row_id"]])
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    y = np.concatenate(targets, axis=0).astype(np.uint8)
    return macro_auc(y, pred), pred, y, row_ids


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 7177)) + int(args.fold))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    meta = pd.read_parquet(cfg["data"]["soundscape_meta"])
    y = np.load(cfg["data"]["soundscape_targets"])
    labels = pd.read_csv(cfg["data"]["labels_csv"])["primary_label"].astype(str).tolist()
    assert y.shape[1] == len(labels)
    assert len(meta) == y.shape[0]

    fold = int(args.fold)
    train_idx = np.where(meta["fold"].to_numpy() != fold)[0]
    valid_idx = np.where(meta["fold"].to_numpy() == fold)[0]
    if args.limit_valid:
        valid_idx = valid_idx[: int(args.limit_valid)]

    train_meta = meta.iloc[train_idx].reset_index(drop=True)
    valid_meta = meta.iloc[valid_idx].reset_index(drop=True)
    train_y = y[train_idx].astype(np.float32)
    valid_y = y[valid_idx]
    train_mask = np.ones_like(train_y, dtype=np.float32)

    train_meta, train_y, train_mask = concat_training_sources(train_meta, train_y, train_mask, cfg, fold)
    if args.limit_train:
        rng = np.random.default_rng(int(cfg.get("seed", 7177)) + fold)
        keep = np.sort(rng.choice(len(train_meta), size=min(int(args.limit_train), len(train_meta)), replace=False))
        train_meta = train_meta.iloc[keep].reset_index(drop=True)
        train_y = train_y[keep]
        train_mask = train_mask[keep]

    print("Train rows by source:", train_meta.get("source", pd.Series(["soundscape"] * len(train_meta))).value_counts().to_dict())
    print("Valid rows:", len(valid_meta))

    train_ds = SoundscapeWindowDataset(train_meta, train_y, cfg, target_masks=train_mask)
    valid_ds = SoundscapeWindowDataset(valid_meta, valid_y, cfg)
    weights = make_sample_weights(train_meta, train_y, cfg)
    train_loader = make_loader(train_ds, cfg, train=True, weights=weights)
    valid_loader = make_loader(valid_ds, cfg, train=False)

    model = build_model(cfg, pretrained=(not args.no_pretrained)).to(device)
    pos = train_y.sum(axis=0).astype(np.float32)
    neg = max(1, len(train_y)) - pos
    pos_weight = np.sqrt(neg / np.maximum(pos, 1.0))
    pos_weight = np.clip(pos_weight, 1.0, float(cfg["train"].get("pos_weight_clip", 20.0)))
    loss_type = str(cfg["train"].get("loss", "bce")).lower()
    bce_criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
    )
    dpo_beta = float(cfg["train"].get("dpo_beta", 1.0))
    dpo_bce_weight = float(cfg["train"].get("dpo_bce_weight", 0.1))
    criterion = bce_criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    epochs = int(args.epochs or cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(cfg["train"].get("min_lr", 1e-6)),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 0.0))

    out_dir = ensure_dir(Path(cfg["output"]["dir"]) / f"fold{fold}")
    with (out_dir / "run_config.json").open("w") as f:
        json.dump({"config": cfg, "fold": fold, "labels": labels}, f, indent=2)

    best_score = -1.0
    history = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        pbar = tqdm(train_loader, desc=f"fold{fold} epoch{epoch}")
        for step, batch in enumerate(pbar, start=1):
            x = batch["image"].to(device, non_blocking=True).float()
            target = batch["target"].to(device, non_blocking=True).float()
            target_mask = batch["target_mask"].to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                if loss_type == "dpo":
                    # DPO: pairwise ranking loss per class
                    dpo_loss = torch.tensor(0.0, device=device)
                    n_pairs = 0
                    for c in range(logits.shape[1]):
                        pos_idx = ((target[:, c] > 0.5) & (target_mask[:, c] > 0.5)).nonzero(as_tuple=True)[0]
                        neg_idx = ((target[:, c] < 0.5) & (target_mask[:, c] > 0.5)).nonzero(as_tuple=True)[0]
                        if len(pos_idx) < 1 or len(neg_idx) < 1:
                            continue
                        n_sample = min(20, len(pos_idx) * len(neg_idx))
                        pi = pos_idx[torch.randint(len(pos_idx), (n_sample,))]
                        ni = neg_idx[torch.randint(len(neg_idx), (n_sample,))]
                        dpo_loss = dpo_loss + (-torch.nn.functional.logsigmoid(
                            dpo_beta * (logits[pi, c] - logits[ni, c])
                        )).mean()
                        n_pairs += 1
                    if n_pairs > 0:
                        dpo_loss = dpo_loss / n_pairs
                    bce_loss = (bce_criterion(logits, target) * target_mask).mean()
                    loss = (dpo_loss + dpo_bce_weight * bce_loss) / grad_accum
                else:
                    loss_raw = criterion(logits, target)
                    loss = (loss_raw * target_mask).mean() / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.detach().cpu()) * grad_accum)
            pbar.set_postfix(loss=np.mean(losses[-20:]))

        scheduler.step()
        score, pred, true, row_ids = validate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_auc": score}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(row)

        if np.isfinite(score) and score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "fold": fold,
                    "labels": labels,
                    "valid_auc": score,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )
            np.save(out_dir / "valid_pred.npy", pred)
            np.save(out_dir / "valid_true.npy", true)
            valid_meta.assign(row_id=row_ids).to_parquet(out_dir / "valid_meta.parquet", index=False)
            np.savez_compressed(
                out_dir / f"fold{fold}_valid_predictions.npz",
                row_id=np.array(row_ids),
                file_id=valid_meta["file_id"].astype(str).to_numpy(),
                filename=valid_meta["filename"].astype(str).to_numpy(),
                window_idx=valid_meta["window_idx"].to_numpy(np.int16),
                y_true=true.astype(np.uint8),
                pred=pred.astype(np.float32),
                labels=np.array(labels),
                fold=np.full(len(row_ids), fold, dtype=np.int16),
            )
            print(f"Saved best checkpoint: {out_dir / 'best.pt'} score={best_score:.6f}")


if __name__ == "__main__":
    main()

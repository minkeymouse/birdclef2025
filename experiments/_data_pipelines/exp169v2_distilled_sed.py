#!/usr/bin/env python3
"""exp169 v2 — Tucker-style distilled SED with random in-file crops.

Differences vs exp169 v1:
  * 3 cached anchors per file (Perch teacher embedding pre-extracted at
    each anchor offset). Training picks one anchor uniformly at random
    each batch.
  * Online ogg decode in DataLoader workers (no waveform cache).
  * 20 epochs (vs v1's 15) since more sample variety should benefit
    longer training.
Everything else is identical to exp169 v1: tf_efficientnet_b0.ns_jft_in1k
backbone, mel-256, GeMFreq + 512 bottleneck + att/cla on h.detach(),
distill_head GAP+Linear -> 1536, loss = 0.5 BCE_clip + 0.5 BCE_frame_max
+ 1.0 MSE distill, AdamW lr=5e-4 wd=1e-4, batch=32, 5-fold StratifiedKFold.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169v2_outputs"
BG_PATH = ROOT / "experiments" / "_data_pipelines" / "exp49_outputs" / "bg_quiet_2025.npz"

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC
N_FFT = 2048; HOP = 512
N_MELS = 256; FMIN = 20; FMAX = 16000
PERCH_DIM = 1536
ALPHA_DISTILL = 1.0
SECONDARY_WEIGHT = 0.3

BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"
N_SPLITS = 5
EPOCHS = 20
BATCH = 32
LR = 5e-4
WD = 1e-4
WARMUP = 1
NUM_WORKERS = 8

GAIN_DB_RANGE = (-6.0, 6.0)
NOISE_SNR_RANGE = (10.0, 30.0)
AUG_PROB = 0.5
BG_MIX_P = 0.3
BG_ALPHA_LO, BG_ALPHA_HI = 0.4, 0.8
SPEC_FREQ_MASK = 24
SPEC_TIME_MASK = 40

EVAL_SS_N_FILES = 11
SEED = 42
DEVICE = "cuda"


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def parse_secondary(x):
    if pd.isna(x) or x in ("[]", ""):
        return []
    try:
        return [s.strip("'\" ") for s in x.strip("[]").split(",") if s.strip("'\" ")]
    except Exception:
        return []


def load_audio(path: Path) -> np.ndarray | None:
    try:
        wav, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != SR:
            import torchaudio.functional as TF
            wav = TF.resample(torch.from_numpy(wav), sr0, SR).numpy()
        return wav.astype(np.float32)
    except Exception:
        return None


# ----------------------------------------------------------------------- data
class CachedDataset(Dataset):
    def __init__(self, indices: np.ndarray, l2i: dict, train: bool,
                 secondary_map: dict, ss_labels_map: dict,
                 primary_idx_full: np.ndarray, is_ss_full: np.ndarray,
                 files: np.ndarray, n_anchors: np.ndarray,
                 anchor_off: np.ndarray, anchor_emb: np.ndarray,
                 bg_pool: np.ndarray | None = None):
        self.indices = indices
        self.l2i = l2i; self.n_cls = len(l2i); self.train = train
        self.secondary_map = secondary_map; self.ss_labels_map = ss_labels_map
        self.primary_idx_full = primary_idx_full; self.is_ss_full = is_ss_full
        self.files = files; self.n_anchors = n_anchors
        self.anchor_off = anchor_off; self.anchor_emb = anchor_emb
        self.bg_pool = bg_pool

    def __len__(self):
        return len(self.indices)

    def _load_clip(self, gi: int, anchor_id: int) -> np.ndarray:
        is_ss = self.is_ss_full[gi]
        sub = "train_soundscapes" if is_ss else "train_audio"
        wav = load_audio(DATA / sub / self.files[gi])
        if wav is None or len(wav) == 0:
            return np.zeros(WIN_SAMPLES, dtype=np.float32)
        if len(wav) < WIN_SAMPLES:
            reps = WIN_SAMPLES // len(wav) + 1
            wav = np.tile(wav, reps)[:WIN_SAMPLES]
            return wav.astype(np.float32)
        off = int(self.anchor_off[gi, anchor_id])
        off = max(0, min(len(wav) - WIN_SAMPLES, off))
        return wav[off:off + WIN_SAMPLES].astype(np.float32)

    def __getitem__(self, idx):
        gi = int(self.indices[idx])
        n_anch = max(1, int(self.n_anchors[gi]))
        if self.train:
            anchor_id = random.randint(0, n_anch - 1)
        else:
            anchor_id = 0
        wav = self._load_clip(gi, anchor_id)
        perch = self.anchor_emb[gi, anchor_id].astype(np.float32)

        y = np.zeros(self.n_cls, dtype=np.float32)
        if self.is_ss_full[gi] == 1:
            for lbl in self.ss_labels_map.get(gi, []):
                if lbl in self.l2i:
                    y[self.l2i[lbl]] = 1.0
            primary_idx = -1; is_ta = 0
        else:
            primary_idx = int(self.primary_idx_full[gi])
            y[primary_idx] = 1.0
            for sl in self.secondary_map.get(gi, []):
                if sl in self.l2i:
                    y[self.l2i[sl]] = SECONDARY_WEIGHT
            is_ta = 1

        if self.train:
            if random.random() < AUG_PROB:
                db = random.uniform(*GAIN_DB_RANGE)
                wav = wav * (10.0 ** (db / 20.0))
            if random.random() < AUG_PROB:
                snr = random.uniform(*NOISE_SNR_RANGE)
                sig_p = float(np.mean(wav ** 2)) + 1e-8
                noise_p = sig_p / (10.0 ** (snr / 10.0))
                wav = wav + np.random.randn(len(wav)).astype(np.float32) * math.sqrt(noise_p)
            if self.bg_pool is not None and random.random() < BG_MIX_P:
                bg = self.bg_pool[np.random.randint(0, len(self.bg_pool))]
                if len(bg) >= WIN_SAMPLES:
                    s = random.randint(0, len(bg) - WIN_SAMPLES)
                    bg = bg[s:s + WIN_SAMPLES]
                else:
                    reps = WIN_SAMPLES // max(1, len(bg)) + 1
                    bg = np.tile(bg, reps)[:WIN_SAMPLES]
                lam = random.uniform(BG_ALPHA_LO, BG_ALPHA_HI)
                wav = lam * wav + (1.0 - lam) * bg.astype(np.float32)
        return (torch.from_numpy(wav.astype(np.float32)),
                torch.from_numpy(perch),
                torch.from_numpy(y),
                int(primary_idx),
                int(is_ta))


# ----------------------------------------------------------------------- model (same as v1)
class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        x = x.float()
        m = self.adb(self.mel(x))
        mean = m.mean(dim=(1, 2), keepdim=True)
        std = m.std(dim=(1, 2), keepdim=True).clamp(min=1e-3)
        m = (m - mean) / std
        return m.unsqueeze(1)


class GeMFreqPool(nn.Module):
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p_init)))
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=2)
        return x.pow(1.0 / self.p)


class SpecAug(nn.Module):
    def __init__(self, f=SPEC_FREQ_MASK, t=SPEC_TIME_MASK):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=f)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=t)

    def forward(self, x):
        return self.tm(self.fm(x))


class DistilledSED(nn.Module):
    def __init__(self, n_cls: int = 234, backbone: str = BACKBONE):
        super().__init__()
        self.mel = MelExtractor()
        self.spec_aug = SpecAug()
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=1,
            drop_rate=0.1, drop_path_rate=0.1,
            num_classes=0, global_pool="",
        )
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
            C = feat.shape[1]
        self.feat_dim = C
        self.gem_freq = GeMFreqPool()
        self.bottleneck = nn.Sequential(
            nn.Linear(C, 512), nn.ReLU(inplace=True), nn.Dropout(0.25),
        )
        self.att = nn.Conv1d(512, n_cls, kernel_size=1)
        self.cla = nn.Conv1d(512, n_cls, kernel_size=1)
        self.distill_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, PERCH_DIM),
        )

    def forward(self, x, training_aug: bool = False):
        m = self.mel(x)
        if training_aug and self.training:
            m = self.spec_aug(m)
        h = self.backbone(m)
        distill_emb = self.distill_head(h)
        h_cls = h.detach()
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.transpose(1, 2)
        h_cls = self.bottleneck(h_cls)
        h_cls = h_cls.transpose(1, 2)
        a = torch.tanh(self.att(h_cls))
        norm_att = torch.softmax(a, dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = (norm_att * framewise_logits).sum(dim=2)
        return clip_logits, framewise_logits, distill_emb


def hybrid_loss(clip_logits, framewise_logits, distill_emb, perch_emb, y):
    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, y)
    frame_max = framewise_logits.max(dim=2).values
    bce_frame = F.binary_cross_entropy_with_logits(frame_max, y)
    cls_loss = 0.5 * bce_clip + 0.5 * bce_frame
    distill_loss = F.mse_loss(distill_emb, perch_emb)
    return cls_loss + ALPHA_DISTILL * distill_loss, bce_clip.item(), bce_frame.item(), distill_loss.item()


def train_epoch(model, loader, opt, dev):
    model.train()
    tot = 0.0; n = 0; nan_skip = 0
    bce_c_sum = 0.0; bce_f_sum = 0.0; mse_sum = 0.0
    for wav, perch, y, _pr, _is_ta in loader:
        wav = wav.to(dev, non_blocking=True)
        perch = perch.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        if not torch.isfinite(wav).all():
            wav = torch.nan_to_num(wav, 0.0, 1.0, -1.0)
        opt.zero_grad(set_to_none=True)
        clip_l, frame_l, demb = model(wav, training_aug=True)
        loss, bc, bf, ms = hybrid_loss(clip_l, frame_l, demb, perch, y)
        if not torch.isfinite(loss):
            nan_skip += 1; continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            nan_skip += 1; opt.zero_grad(set_to_none=True); continue
        opt.step()
        tot += loss.item() * wav.size(0); n += wav.size(0)
        bce_c_sum += bc; bce_f_sum += bf; mse_sum += ms
    nb = max(1, len(loader))
    return tot / max(n, 1), bce_c_sum / nb, bce_f_sum / nb, mse_sum / nb, nan_skip


@torch.no_grad()
def evaluate_via_loader(model, ds: CachedDataset, dev, batch=32, max_n=2000):
    model.eval()
    n = min(max_n, len(ds))
    rng = np.random.RandomState(123)
    pick = rng.choice(len(ds), n, replace=False) if len(ds) > n else np.arange(len(ds))
    n_cls = ds.n_cls
    preds = np.zeros((n, n_cls), dtype=np.float32)
    Y = np.zeros((n, n_cls), dtype=np.uint8)
    wavs_buf = []; ys_buf = []; idx_buf = []
    for i, k in enumerate(pick):
        wav, _, y, _pr, _ = ds[int(k)]
        wavs_buf.append(wav); ys_buf.append(y); idx_buf.append(i)
        if len(wavs_buf) >= batch or i + 1 == n:
            x = torch.stack(wavs_buf).to(dev)
            clip_l, _, _ = model(x)
            p = torch.sigmoid(clip_l).cpu().numpy()
            for j, ri in enumerate(idx_buf):
                preds[ri] = p[j]; Y[ri] = ys_buf[j].numpy()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    primary, l2i = build_primaries()
    n_cls = len(l2i)

    print("loading anchor cache...", flush=True)
    cache = np.load(CACHE_DIR / "anchors.npz", allow_pickle=True)
    files = cache["files"]; is_ss = cache["is_ss"]
    ss_endsec = cache["ss_endsec"]; primary_idx = cache["primary_idx"]
    ss_label_str = cache["ss_label_str"]
    n_anchors = cache["n_anchors"]; anchor_off = cache["anchor_off"]
    anchor_emb = cache["anchor_emb"]
    print(f"  files={len(files)}  anchors per file (mean): {n_anchors.mean():.2f}", flush=True)

    ta_df = pd.read_csv(DATA / "train.csv")
    ta_df = ta_df[ta_df["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    ta_df["secondary_list"] = ta_df["secondary_labels"].apply(parse_secondary)
    fname_to_secondary = dict(zip(ta_df["filename"], ta_df["secondary_list"]))
    secondary_map = {int(gi): fname_to_secondary.get(files[gi], [])
                     for gi in np.where(is_ss == 0)[0]}

    ss_labels_map: dict[int, list[str]] = {}
    for gi in np.where(is_ss == 1)[0]:
        s = ss_label_str[gi]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            ss_labels_map[int(gi)] = []
        else:
            ss_labels_map[int(gi)] = [t for t in str(s).split(";") if t.strip()]

    ss_files = sorted({files[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(SEED); rng.shuffle(ss_files)
    eval_files = set(ss_files[:EVAL_SS_N_FILES])
    ss_train_indices = np.array(
        [i for i in np.where(is_ss == 1)[0] if files[i] not in eval_files], dtype=np.int64)
    ss_eval_indices = np.array(
        [i for i in np.where(is_ss == 1)[0] if files[i] in eval_files], dtype=np.int64)
    print(f"SS train: {len(ss_train_indices)} | SS eval: {len(ss_eval_indices)}", flush=True)

    ta_indices = np.where(is_ss == 0)[0]
    ta_labels = primary_idx[ta_indices]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(ta_indices, ta_labels))

    bg_pool = None
    if BG_PATH.exists():
        bg_pool = np.load(BG_PATH)["windows"]
        print(f"BG pool: {bg_pool.shape}", flush=True)

    folds_to_run = list(range(N_SPLITS)) if args.fold < 0 else [args.fold]
    for fold_id in folds_to_run:
        print(f"\n========== FOLD {fold_id} ==========", flush=True)
        tr_local, va_local = folds[fold_id]
        tr_global = ta_indices[tr_local]
        va_global = ta_indices[va_local]
        train_idx = np.concatenate([tr_global, ss_train_indices])
        print(f"train_idx {len(train_idx)} (TA {len(tr_global)} + SS {len(ss_train_indices)})", flush=True)

        ds_train = CachedDataset(
            train_idx, l2i, True, secondary_map, ss_labels_map,
            primary_idx, is_ss, files, n_anchors, anchor_off, anchor_emb, bg_pool,
        )
        weights = np.ones(len(train_idx), dtype=np.float64)
        ss_w = (len(tr_global) / max(1, len(ss_train_indices))) * (0.10 / 0.90)
        weights[len(tr_global):] = ss_w
        sampler = WeightedRandomSampler(weights, num_samples=len(tr_global), replacement=True)
        loader = DataLoader(
            ds_train, batch_size=args.batch, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=True,
        )

        ds_val_ta = CachedDataset(
            va_global, l2i, False, secondary_map, ss_labels_map,
            primary_idx, is_ss, files, n_anchors, anchor_off, anchor_emb, None,
        )
        ds_val_ss = CachedDataset(
            ss_eval_indices, l2i, False, secondary_map, ss_labels_map,
            primary_idx, is_ss, files, n_anchors, anchor_off, anchor_emb, None,
        )

        model = DistilledSED(n_cls=n_cls).to(DEVICE)
        if fold_id == folds_to_run[0]:
            print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - WARMUP), eta_min=1e-5)

        fold_dir = CACHE_DIR / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best = {"val_TA": -1.0, "epoch": 0}
        for ep in range(1, args.epochs + 1):
            if ep <= WARMUP:
                lr_now = LR * ep / WARMUP
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            t0 = time.time()
            tr_loss, bc, bf, ms, nans = train_epoch(model, loader, opt, DEVICE)
            if ep > WARMUP:
                sched.step()
            cur_lr = opt.param_groups[0]["lr"]
            ta_auc, n_ta = evaluate_via_loader(model, ds_val_ta, DEVICE, max_n=1500)
            ss_auc, n_ss = evaluate_via_loader(model, ds_val_ss, DEVICE, max_n=200)
            dt = time.time() - t0
            row = dict(epoch=ep, lr=cur_lr, loss=tr_loss, bce_clip=bc, bce_frame=bf, mse=ms,
                       val_TA=ta_auc, n_TA=n_ta, val_SS=ss_auc, n_SS=n_ss, time_s=dt, nan_skip=nans)
            history.append(row)
            print(f"  ep{ep:02d} lr {cur_lr:.5f}  loss {tr_loss:.4f} (bce_c {bc:.3f} bce_f {bf:.3f} mse {ms:.3f})  "
                  f"val_TA {ta_auc:.4f} ({n_ta})  val_SS {ss_auc:.4f} ({n_ss})  nan {nans}  ({dt:.0f}s)",
                  flush=True)
            if ta_auc > best["val_TA"]:
                best = {"val_TA": ta_auc, "val_SS": ss_auc, "epoch": ep}
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": ep,
                     "val_TA": ta_auc, "val_SS": ss_auc,
                     "config": dict(backbone=BACKBONE, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                    seed=args.seed, fold=fold_id, k_anchors=int(n_anchors.max()))},
                    fold_dir / "best_ckpt.pt",
                )
        with open(fold_dir / "history.json", "w") as f:
            json.dump({"history": history, "best": best,
                       "config": dict(backbone=BACKBONE, fold=fold_id, epochs=args.epochs,
                                      batch=args.batch, lr=LR, alpha_distill=ALPHA_DISTILL,
                                      seed=args.seed, k_anchors=int(n_anchors.max()))},
                      f, indent=2, default=float)
        print(f"FOLD {fold_id} done. best ep {best['epoch']}  val_TA {best['val_TA']:.4f}  "
              f"val_SS {best['val_SS']:.4f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""exp169 — Tucker-style distilled SED, full 5-fold training.

Recipe (matches tuckerarrants/bc2026-distilled-sed):
  backbone:    tf_efficientnet_b0.ns_jft_in1k (NoisyStudent JFT-300M)
  spec:        n_mels=256, fmin=20, fmax=16000, per-spec z-score
  heads:
    SED:       GeMFreq -> bottleneck -> att+cla on h.detach()
    distill:   GAP -> Linear(C->1536) on h (gradient flows here)
  loss:        0.5*BCE_clip + 0.5*BCE_frame_max + 1.0*MSE(distill_emb, perch_emb)

Per-fold training reads pre-cached (waveform, perch_emb) pairs from
exp169_perch_cache.py. Online augmentation (gain, noise, BG mix, SpecAugment)
is applied to the student input; distillation target stays clean Perch.

Output:
  experiments/_data_pipelines/exp169_outputs/
    fold{0..4}/best_ckpt.pt  best by val_TA
    fold{0..4}/history.json
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"
BG_PATH = ROOT / "experiments" / "_data_pipelines" / "exp49_outputs" / "bg_quiet_2025.npz"
OUT = CACHE_DIR

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC                              # 160000
N_FFT = 2048
HOP = 512
N_MELS = 256
FMIN = 20
FMAX = 16000
PERCH_DIM = 1536
ALPHA_DISTILL = 1.0
SECONDARY_WEIGHT = 0.3

BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"
N_SPLITS = 5
EPOCHS = 15
BATCH = 32
LR = 5e-4
WD = 1e-4
WARMUP = 1
NUM_WORKERS = 6

# Augmentation
GAIN_DB_RANGE = (-6.0, 6.0)
NOISE_SNR_RANGE = (10.0, 30.0)
AUG_PROB = 0.5
BG_MIX_P = 0.3
BG_ALPHA_LO, BG_ALPHA_HI = 0.4, 0.8
SPEC_FREQ_MASK = 24
SPEC_TIME_MASK = 40
SPEC_AUG_P = 0.5

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


# ----------------------------------------------------------------------- data
class CachedDataset(Dataset):
    """Reads from waveforms_fp16.npy / perch_emb.npy / meta.npz (memmap)."""

    def __init__(self, indices: np.ndarray, l2i: dict, train: bool, secondary_map: dict,
                 ss_labels_map: dict, primary_idx_full: np.ndarray, is_ss_full: np.ndarray,
                 wav_mm: np.ndarray, perch_mm: np.ndarray, bg_pool: np.ndarray | None = None):
        self.indices = indices
        self.l2i = l2i
        self.n_cls = len(l2i)
        self.train = train
        self.secondary_map = secondary_map
        self.ss_labels_map = ss_labels_map
        self.primary_idx_full = primary_idx_full
        self.is_ss_full = is_ss_full
        self.wav_mm = wav_mm
        self.perch_mm = perch_mm
        self.bg_pool = bg_pool

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gi = int(self.indices[idx])
        wav = self.wav_mm[gi].astype(np.float32)
        perch = self.perch_mm[gi].astype(np.float32)

        y = np.zeros(self.n_cls, dtype=np.float32)
        if self.is_ss_full[gi] == 1:
            for lbl in self.ss_labels_map.get(gi, []):
                if lbl in self.l2i:
                    y[self.l2i[lbl]] = 1.0
            primary_idx = -1
            is_ta = 0
        else:
            primary_idx = int(self.primary_idx_full[gi])
            y[primary_idx] = 1.0
            for sl in self.secondary_map.get(gi, []):
                if sl in self.l2i:
                    y[self.l2i[sl]] = SECONDARY_WEIGHT
            is_ta = 1

        if self.train:
            # gain
            if random.random() < AUG_PROB:
                db = random.uniform(*GAIN_DB_RANGE)
                wav = wav * (10.0 ** (db / 20.0))
            # additive gaussian noise
            if random.random() < AUG_PROB:
                snr = random.uniform(*NOISE_SNR_RANGE)
                sig_p = np.mean(wav ** 2) + 1e-8
                noise_p = sig_p / (10.0 ** (snr / 10.0))
                wav = wav + np.random.randn(len(wav)).astype(np.float32) * math.sqrt(noise_p)
            # 2025 BG additive
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


# ----------------------------------------------------------------------- model
class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):  # x: (B, T)
        # Always run mel + dB in fp32 to avoid log10/clamp underflow.
        x = x.float()
        m = self.adb(self.mel(x))                                # (B, M, T')
        # per-spec z-score
        mean = m.mean(dim=(1, 2), keepdim=True)
        std = m.std(dim=(1, 2), keepdim=True).clamp(min=1e-3)
        m = (m - mean) / std
        return m.unsqueeze(1)                                    # (B, 1, M, T')


class GeMFreqPool(nn.Module):
    """GeM pooling along frequency axis only (M dim of (B, C, M, T))."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p_init)))
        self.eps = eps

    def forward(self, x):                                        # (B, C, M, T)
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=2)                                        # pool over freq
        return x.pow(1.0 / self.p)                               # (B, C, T)


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
        # distillation head: GAP -> Linear(C -> 1536)
        self.distill_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, PERCH_DIM),
        )

    def forward(self, x, training_aug: bool = False):
        m = self.mel(x)                                          # (B, 1, M, T')
        if training_aug and self.training:
            m = self.spec_aug(m)
        h = self.backbone(m)                                     # (B, C, m', t')
        # ---- distillation branch — gradient flows ----
        distill_emb = self.distill_head(h)                       # (B, 1536)
        # ---- SED branch — stop gradient ----
        h_cls = h.detach()
        h_cls = self.gem_freq(h_cls)                             # (B, C, T)
        h_cls = h_cls.transpose(1, 2)                            # (B, T, C)
        h_cls = self.bottleneck(h_cls)                           # (B, T, 512)
        h_cls = h_cls.transpose(1, 2)                            # (B, 512, T)
        a = torch.tanh(self.att(h_cls))
        norm_att = torch.softmax(a, dim=-1)                      # softmax over time
        framewise_logits = self.cla(h_cls)                       # (B, n_cls, T)
        clip_logits = (norm_att * framewise_logits).sum(dim=2)   # (B, n_cls)
        return clip_logits, framewise_logits, distill_emb


# ----------------------------------------------------------------------- loss
def hybrid_loss(clip_logits, framewise_logits, distill_emb, perch_emb, y):
    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, y)
    frame_max = framewise_logits.max(dim=2).values                # (B, n_cls)
    bce_frame = F.binary_cross_entropy_with_logits(frame_max, y)
    cls_loss = 0.5 * bce_clip + 0.5 * bce_frame
    distill_loss = F.mse_loss(distill_emb, perch_emb)
    return cls_loss + ALPHA_DISTILL * distill_loss, bce_clip.item(), bce_frame.item(), distill_loss.item()


# ------------------------------------------------------------------- training
def train_epoch(model, loader, opt, scaler, dev):
    """Full fp32 (AMP disabled — fp16 mel/db underflows produced 70% NaN batches in v1).

    The mel pipeline (MelSpectrogram + AmplitudeToDB) clamps to amin=1e-10 then
    takes log10, which underflows in fp16 and propagates NaN through the rest
    of the network. We keep everything in fp32 for stability; an EfficientNet-B0
    forward+backward at batch=32, mel-256 fits comfortably in 32GB RTX 5090.
    """
    del scaler  # unused; kept for signature compatibility
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
            nan_skip += 1
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            nan_skip += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        tot += loss.item() * wav.size(0); n += wav.size(0)
        bce_c_sum += bc; bce_f_sum += bf; mse_sum += ms
    nb = max(1, len(loader))
    return tot / max(n, 1), bce_c_sum / nb, bce_f_sum / nb, mse_sum / nb, nan_skip


@torch.no_grad()
def evaluate_ta(model, indices, wav_mm, primary_idx_full, n_cls, dev, batch=32, max_n=2000):
    model.eval()
    if len(indices) > max_n:
        rng = np.random.RandomState(123)
        indices = rng.choice(indices, max_n, replace=False)
    preds = np.zeros((len(indices), n_cls), dtype=np.float32)
    Y = np.zeros((len(indices), n_cls), dtype=np.uint8)
    for i in range(0, len(indices), batch):
        sl = indices[i:i + batch]
        wavs = wav_mm[sl].astype(np.float32)
        x = torch.from_numpy(wavs).to(dev)
        clip_l, _, _ = model(x)
        p = torch.sigmoid(clip_l).float().cpu().numpy()
        preds[i:i + len(sl)] = p
        for k, gi in enumerate(sl):
            Y[i + k, int(primary_idx_full[gi])] = 1
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


@torch.no_grad()
def evaluate_ss(model, ss_indices, wav_mm, ss_labels_map, l2i, dev, batch=32):
    model.eval()
    n_cls = len(l2i)
    preds = np.zeros((len(ss_indices), n_cls), dtype=np.float32)
    Y = np.zeros((len(ss_indices), n_cls), dtype=np.uint8)
    for i in range(0, len(ss_indices), batch):
        sl = ss_indices[i:i + batch]
        wavs = wav_mm[sl].astype(np.float32)
        x = torch.from_numpy(wavs).to(dev)
        clip_l, _, _ = model(x)
        p = torch.sigmoid(clip_l).float().cpu().numpy()
        preds[i:i + len(sl)] = p
        for k, gi in enumerate(sl):
            for lbl in ss_labels_map.get(int(gi), []):
                if lbl in l2i:
                    Y[i + k, l2i[lbl]] = 1
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
    parser.add_argument("--fold", type=int, default=-1, help="train one fold (0..4); -1 = all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    primary, l2i = build_primaries()
    n_cls = len(l2i)

    print("loading cache (memmap)...", flush=True)
    wav_mm = np.load(CACHE_DIR / "waveforms_fp16.npy", mmap_mode="r")
    perch_mm = np.load(CACHE_DIR / "perch_emb.npy", mmap_mode="r")
    meta = np.load(CACHE_DIR / "meta.npz", allow_pickle=True)
    filenames = meta["filenames"]
    is_ss_full = meta["is_ss"]
    primary_idx_full = meta["primary_idx"]
    ss_label_str = meta["ss_label_str"]
    print(f"  waveforms shape {wav_mm.shape}  perch shape {perch_mm.shape}", flush=True)

    # Build secondary map for TA rows from train.csv
    ta_df = pd.read_csv(DATA / "train.csv")
    ta_df = ta_df[ta_df["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    ta_df["secondary_list"] = ta_df["secondary_labels"].apply(parse_secondary)
    fname_to_secondary = dict(zip(ta_df["filename"], ta_df["secondary_list"]))
    secondary_map: dict[int, list[str]] = {}
    for gi in np.where(is_ss_full == 0)[0]:
        secondary_map[int(gi)] = fname_to_secondary.get(filenames[gi], [])

    # Build SS labels map
    ss_labels_map: dict[int, list[str]] = {}
    for gi in np.where(is_ss_full == 1)[0]:
        s = ss_label_str[gi]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            ss_labels_map[int(gi)] = []
        else:
            ss_labels_map[int(gi)] = [t for t in str(s).split(";") if t.strip()]

    # SS train/eval split (file-level)
    ss_files = sorted({filenames[i] for i in np.where(is_ss_full == 1)[0]})
    rng = np.random.RandomState(SEED)
    rng.shuffle(ss_files)
    ss_eval_files = set(ss_files[:EVAL_SS_N_FILES])
    ss_train_indices = np.array(
        [i for i in np.where(is_ss_full == 1)[0] if filenames[i] not in ss_eval_files],
        dtype=np.int64,
    )
    ss_eval_indices = np.array(
        [i for i in np.where(is_ss_full == 1)[0] if filenames[i] in ss_eval_files],
        dtype=np.int64,
    )
    print(f"SS train: {len(ss_train_indices)} | SS eval: {len(ss_eval_indices)}", flush=True)

    # 5-fold StratifiedKFold on TA
    ta_indices = np.where(is_ss_full == 0)[0]
    ta_labels = primary_idx_full[ta_indices]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(ta_indices, ta_labels))
    print(f"folds: {N_SPLITS} (TA only)  total TA {len(ta_indices)}", flush=True)

    # 2025 BG pool
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
        print(f"train_idx {len(train_idx)} (TA {len(tr_global)} + SS {len(ss_train_indices)})  "
              f"val_TA {len(va_global)}  val_SS {len(ss_eval_indices)}", flush=True)

        ds_train = CachedDataset(
            train_idx, l2i, train=True, secondary_map=secondary_map,
            ss_labels_map=ss_labels_map, primary_idx_full=primary_idx_full,
            is_ss_full=is_ss_full, wav_mm=wav_mm, perch_mm=perch_mm, bg_pool=bg_pool,
        )
        # SS has fewer samples; weight ss to ~10% of batch
        weights = np.ones(len(train_idx), dtype=np.float64)
        ss_w = (len(tr_global) / max(1, len(ss_train_indices))) * (0.10 / 0.90)
        weights[len(tr_global):] = ss_w
        sampler = WeightedRandomSampler(weights, num_samples=len(tr_global), replacement=True)
        loader = DataLoader(
            ds_train, batch_size=args.batch, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=True,
        )

        model = DistilledSED(n_cls=n_cls).to(DEVICE)
        if fold_id == folds_to_run[0]:
            print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - WARMUP), eta_min=1e-5)
        scaler = None  # AMP disabled — fp16 mel underflow caused NaN cascade in v1

        fold_dir = OUT / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best = {"val_TA": -1.0, "epoch": 0}
        for ep in range(1, args.epochs + 1):
            if ep <= WARMUP:
                lr_now = LR * ep / WARMUP
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            t0 = time.time()
            tr_loss, bc, bf, ms, nans = train_epoch(model, loader, opt, scaler, DEVICE)
            if ep > WARMUP:
                sched.step()
            cur_lr = opt.param_groups[0]["lr"]
            ta_auc, n_ta = evaluate_ta(model, va_global, wav_mm, primary_idx_full, n_cls, DEVICE)
            ss_auc, n_ss = evaluate_ss(model, ss_eval_indices, wav_mm, ss_labels_map, l2i, DEVICE)
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
                                    seed=args.seed, fold=fold_id)},
                    fold_dir / "best_ckpt.pt",
                )
        with open(fold_dir / "history.json", "w") as f:
            json.dump({"history": history, "best": best,
                       "config": dict(backbone=BACKBONE, fold=fold_id, epochs=args.epochs,
                                      batch=args.batch, lr=LR, alpha_distill=ALPHA_DISTILL,
                                      seed=args.seed)},
                      f, indent=2, default=float)
        print(f"FOLD {fold_id} done. best ep {best['epoch']}  val_TA {best['val_TA']:.4f}  "
              f"val_SS {best['val_SS']:.4f}", flush=True)


if __name__ == "__main__":
    main()

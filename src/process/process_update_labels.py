#!/usr/bin/env python3
"""
process_update_labels.py ─ refresh soft labels for soundscapes & mix-ups
------------------------------------------------------------------------
* rows with source ∈ {"soundscapes","mixup_audio"} are updated in-place
* HARD ensemble inside each architecture:
      prob_arch = min(prob_run1, …, prob_runN)
* then simple average across architectures
* Apply Kalman‐filter update to fuse old labels (state) + ensemble (measurement)
* supports --q, --r, --device, --batch, --dry-run
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import torchvision     # for EfficientNet
import timm            # for RegNetY

# ──────────────────── CLI ────────────────────
p = argparse.ArgumentParser()
p.add_argument("--q",         type=float, default=1e-4,
               help="process‐noise variance (Q)")
p.add_argument("--r",         type=float, default=1e-3,
               help="measurement‐noise variance (R)")
p.add_argument("--device",    default="auto", help="cpu | cuda | auto")
p.add_argument("--batch",     type=int,   default=32,  help="inference batch size")
p.add_argument("--dry-run",   action="store_true")
args = p.parse_args()

# ──────────────── config & logging ───────────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "process.yaml") as f:
    CFG = yaml.safe_load(f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("update_labels")

# pick device
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
log.info("Device=%s | Q=%.1e | R=%.1e", device, args.q, args.r)

# ────────────────── data subset ───────────────
META   = ROOT / CFG["paths"]["train_metadata"]
meta   = pd.read_csv(META)
TARGET = {"soundscapes", "mixup_audio"}
sel    = meta[meta["source"].isin(TARGET)]
if sel.empty:
    log.info("Nothing to update – exiting.")
    sys.exit()

h, w = CFG["mel"]["target_shape"]

class MelSet(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        m = np.load(r.mel_path).astype(np.float32)
        if m.shape != (h, w):
            from src.utils import utils
            m = utils.resize_mel(m, h, w)
        x = torch.from_numpy(m).unsqueeze(0).repeat(3,1,1)
        return x, r.label_path

loader = DataLoader(
    MelSet(sel),
    batch_size=args.batch,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type=="cuda"),
)

# ─────────────── build ensemble ───────────────
tax_csv = CFG["paths"]["taxonomy_csv"]
n_cls   = len(pd.read_csv(tax_csv))

def init_model(arch: str, n_cls: int):
    """Instantiate exactly as in training for each architecture."""
    if arch == "efficientnet_b0":
        net = torchvision.models.efficientnet_b0(weights=None)
        in_f = net.classifier[1].in_features
        net.classifier[1] = torch.nn.Linear(in_f, n_cls)
    elif arch == "regnety_008":
        net = timm.create_model(
            "regnety_008",
            pretrained=False,
            in_chans=3,
            num_classes=n_cls,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
    else:
        raise ValueError(f"Unsupported arch '{arch}'")
    return net

MODEL_DIR = ROOT / CFG["paths"]["models_dir"]
ckpts = sorted(MODEL_DIR.glob("*.pth"))
if not ckpts:
    log.error("No checkpoints found in %s", MODEL_DIR)
    sys.exit(1)

# group runs by architecture
arch2models: Dict[str, List[torch.nn.Module]] = {}
for ck in ckpts:
    if "efficientnet_b0" in ck.name:
        arch = "efficientnet_b0"
    elif "regnety_008" in ck.name:
        arch = "regnety_008"
    else:
        log.warning("Skipping unknown checkpoint: %s", ck.name)
        continue

    mdl = init_model(arch, n_cls)
    state = torch.load(ck, map_location="cpu")
    sd = state.get("model_state_dict", state)
    missing, unexpected = mdl.load_state_dict(sd, strict=False)
    log.info("Loaded %s (%s): missing=%d, unexpected=%d",
             ck.name, arch, len(missing), len(unexpected))
    mdl.to(device).eval()
    arch2models.setdefault(arch, []).append(mdl)

# sanity check
for arch in ("efficientnet_b0","regnety_008"):
    if arch not in arch2models:
        log.error("Need at least one %s checkpoint; found none", arch)
        sys.exit(1)

# ─────────── setup Kalman parameters ───────────
# We'll keep a small per‐class covariance vector for each label file,
# saving it alongside the .npy so future runs can continue filtering.
Q = args.q   # process noise variance
R = args.r   # measurement noise variance

def load_covariance(lp: Path) -> np.ndarray:
    cov_path = lp.with_name(lp.stem + "_cov.npy")
    if cov_path.exists():
        return np.load(cov_path)
    else:
        # initialize large uncertainty so first measurement is trusted
        return np.ones(n_cls, dtype=np.float32)

def save_covariance(lp: Path, P: np.ndarray):
    cov_path = lp.with_name(lp.stem + "_cov.npy")
    np.save(cov_path, P)

# ─────────────── update loop ────────────────
updated = 0
with torch.no_grad():
    for X, paths in loader:
        X = X.to(device, non_blocking=True)

        # 1) HARD ensemble per-architecture (min over runs)
        probs_arch: Dict[str, torch.Tensor] = {}
        for arch, models in arch2models.items():
            runs = torch.stack([torch.softmax(m(X), dim=1) for m in models])
            probs_arch[arch] = runs.min(dim=0).values  # B×C

        # 2) average across architectures
        ensemble = torch.stack(list(probs_arch.values()), dim=0).mean(dim=0)
        ensemble_np = ensemble.cpu().numpy()         # B×C

        # 3) for each example, Kalman‐filter update
        for p_meas, label_path in zip(ensemble_np, paths):
            lp = Path(label_path)
            if not lp.exists():
                log.warning("Missing label file %s", lp)
                continue

            # load prior state & covariance
            x_prev = np.load(lp).astype(np.float32)           # prior
            P_prev = load_covariance(lp)                      # shape (C,)

            # ► PREDICT
            x_pred = x_prev                                   # identity dynamic
            P_pred = P_prev + Q                               # elementwise

            # ► UPDATE
            Kf = P_pred / (P_pred + R + 1e-12)                # gain vector
            x_upd = x_pred + Kf * (p_meas - x_pred)           # updated state
            P_upd = (1.0 - Kf) * P_pred                       # updated covariance

            # clamp to valid probability range
            x_upd = np.clip(x_upd, 0.0, 1.0)

            # save new labels & covariance
            if not args.dry_run:
                np.save(lp, x_upd)
                save_covariance(lp, P_upd)

            updated += 1

log.info("%sUpdated %d label vectors.", 
         "[dry-run] " if args.dry_run else "", updated)

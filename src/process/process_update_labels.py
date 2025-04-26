#!/usr/bin/env python3
"""
process_update_labels.py – Refresh soft‑label vectors using an ensemble and
optional Kalman smoothing
==========================================================================

* Reads **DATABASE/train_metadata.csv** and selects rows whose `source` is
  in {"soundscapes", "mixup_audio"} (the noisy or synthetic sets we want to
  refine).
* Runs the EfficientNet‑B0 and RegNetY‑800MF ensembles on the corresponding
  **pre‑computed mel‑spectrograms** (`mel_path`).
    • For each architecture the element‑wise **minimum** across its three
      checkpoints is taken ("hard" ensembling).
    • The final probability vector is the **mean** of the two architecture
      minima, matching the competition’s reference inference scheme.
* Performs an **element‑wise Kalman update** of the stored probability vector
  using the new ensemble estimate:

    pₜ  ←  pₜ₋₁  +  K ⊙ (z  −  pₜ₋₁) ,   K = (P+Q) / (P+Q+R)

  with diagonal covariance (independent species).  Hyper‑parameters `Q` and
  `R` are set conservatively so the filter moves slowly toward the ensemble
  unless the old label is very uncertain.
* Overwrites the existing `.npy` label file **in‑place**; metadata rows are
  untouched because their paths stay the same.
* Supports `--dry-run` to preview the number of labels that would be updated
  without actually writing to disk.

Run
---
```bash
python -m src.process.process_update_labels               # update labels
python -m src.process.process_update_labels --dry-run     # preview only
```
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# -----------------------------------------------------------------------------
# Configuration & paths
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
config_path  = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils import utils  # noqa: E402  (after sys.path hack)

with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg   = CFG["paths"]
mel_cfg     = CFG["mel"]
label_cfg   = CFG["labeling"]
inf_cfg     = CFG.get("inference",      {})      # optional section
ens_cfg     = CFG.get("ensemble",       {})

MODEL_DIR   = project_root / paths_cfg.get("models_dir", "models")
META_CSV    = project_root / paths_cfg["train_metadata"]

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Update label vectors using ensemble predictions")
parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
parser.add_argument("--device",  default="auto",     help="cpu | cuda | auto (default)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_update_labels")

device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "auto" else torch.device(args.device)
)
log.info("Using device: %s", device)

auto_mixed_sources = {"soundscapes", "mixup_audio"}

# -----------------------------------------------------------------------------
# Dataset – returns (tensor, label_path)
# -----------------------------------------------------------------------------
class MelLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_shape: Tuple[int, int]):
        self.df = df.reset_index(drop=True)
        self.h, self.w = input_shape
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"]).astype(np.float32)
        if mel.shape != (self.h, self.w):
            mel = utils.resize_mel(mel, self.h, self.w)
        tens = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1)  # C=3
        return tens, row["label_path"]

# -----------------------------------------------------------------------------
# Load taxonomy & metadata
# -----------------------------------------------------------------------------
class_list, _ = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), None)
N_CLASSES     = len(class_list)

meta = pd.read_csv(META_CSV)
sel  = meta[meta["source"].isin(auto_mixed_sources)]
if sel.empty:
    log.info("No rows with source in %s – nothing to update.", auto_mixed_sources)
    sys.exit(0)
log.info("%d label files selected for refresh", len(sel))

mel_h, mel_w = tuple(mel_cfg["target_shape"])
loader = DataLoader(MelLabelDataset(sel, (mel_h, mel_w)), batch_size=inf_cfg.get("batch_size", 32), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def _build_model(arch: str, n_classes: int) -> torch.nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n_classes)
        return m
    if arch == "regnety_800mf":
        m = models.regnet_y_800mf(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
        return m
    raise ValueError(f"Unsupported architecture: {arch}")


def _load_checkpoint(path: Path, n_classes: int) -> Tuple[str, torch.nn.Module]:
    arch = "efficientnet_b0" if "efficientnet" in path.name else "regnety_800mf"
    model = _build_model(arch, n_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.to(device).eval()
    return arch, model

# Collect checkpoints according to ensemble patterns
ckpt_patterns = ens_cfg.get("checkpoints", ["efficientnet_b0_run*.pth", "regnety_800mf_run*.pth"])
ckpt_paths: List[Path] = []
for pat in ckpt_patterns:
    ckpt_paths.extend(sorted(MODEL_DIR.glob(pat)))
if not ckpt_paths:
    log.error("No checkpoints found under %s", MODEL_DIR)
    sys.exit(1)

arch_to_models: Dict[str, List[torch.nn.Module]] = {"efficientnet_b0": [], "regnety_800mf": []}
for p in ckpt_paths:
    arch, mdl = _load_checkpoint(p, N_CLASSES)
    arch_to_models[arch].append(mdl)
    log.info("Loaded %s", p.name)

# -----------------------------------------------------------------------------
# Kalman filter implementation (diagonal, independent per class)
# -----------------------------------------------------------------------------
class KalmanProbFilter:
    def __init__(self, n: int, q: float = 1e-4, r: float = 1e-2):
        self.n  = n
        self.Q  = q * np.ones(n, dtype=np.float32)
        self.R  = r * np.ones(n, dtype=np.float32)
        self.P  = np.ones(n, dtype=np.float32)
    def update(self, prior: np.ndarray, meas: np.ndarray) -> np.ndarray:
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        post   = prior + K * (meas - prior)
        self.P = (1. - K) * P_pred
        return post.clip(0., 1.)

kf = KalmanProbFilter(N_CLASSES)

# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------
updated = 0
with torch.no_grad():
    for X, label_path in loader:
        X = X.to(device, non_blocking=True)
        # Gather per‑arch minima
        per_arch_min = {}
        for arch, models in arch_to_models.items():
            preds = [torch.softmax(mdl(X), dim=1) for mdl in models]
            per_arch_min[arch] = torch.min(torch.stack(preds), dim=0).values  # (B, C)
        combined = 0.5 * (per_arch_min["efficientnet_b0"] + per_arch_min["regnety_800mf"])  # (B,C)
        combined = combined.cpu().numpy()

        for i in range(X.size(0)):
            lp = Path(label_path[i])
            if not lp.is_file():
                log.warning("Label file missing: %s", lp)
                continue
            old = np.load(lp)
            new = kf.update(old, combined[i])
            if not args.dry_run:
                np.save(lp, new.astype(np.float32))
            updated += 1

log.info("%sUpdated %d label vectors", "[dry‑run] " if args.dry_run else "", updated)

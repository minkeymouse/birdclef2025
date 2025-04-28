#!/usr/bin/env python3
"""
process_update_labels.py ─ refresh soft labels for soundscapes & mix-ups
------------------------------------------------------------------------
* rows with source ∈ {"soundscapes","mixup_audio"} are updated in-place
* HARD ensemble inside each architecture:
      prob_arch = min (prob_run1, …, prob_runN)
* then simple average across architectures
* fixed-gain blend:  new = (1-k)·old + k·ensemble   (default k = 0.6)
  (behaves like a per-chunk Kalman filter with constant K)
* supports --k, --dry-run, --device
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch, yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# ──────────────────── CLI ────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=float, default=0.6, help="blend factor 0<k≤1")
parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
parser.add_argument("--batch",  type=int, default=32, help="inference batch size")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
K = max(1e-6, min(args.k, 1.0))

# ───────────────── config & logging ───────────
ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "process.yaml") as f:
    CFG = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("update_labels")

device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
          if args.device == "auto" else torch.device(args.device))
log.info("Device: %s | blend k = %.2f", device, K)

# ────────────────── data subset ───────────────
META = ROOT / CFG["paths"]["train_metadata"]
meta = pd.read_csv(META)
TARGET = {"soundscapes", "mixup_audio"}
sel = meta[meta["source"].isin(TARGET)]
if sel.empty:
    log.info("Nothing to update – exiting."); sys.exit()

h, w = CFG["mel"]["target_shape"]

class MelSet(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        m = np.load(r.mel_path).astype(np.float32)
        if m.shape != (h, w):
            from src.utils import utils
            m = utils.resize_mel(m, h, w)
        x = torch.from_numpy(m).unsqueeze(0).repeat(3,1,1)  # C=3
        return x, r.label_path

loader = DataLoader(MelSet(sel), args.batch, False, num_workers=0,
                    pin_memory=(device.type == "cuda"))

# ─────────────── build ensemble ───────────────
tax_csv = CFG["paths"]["taxonomy_csv"]
n_cls = len(pd.read_csv(tax_csv))

def init_model(arch: str):
    if arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, n_cls)
    else:
        net = models.regnet_y_800mf(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, n_cls)
    return net

MODEL_DIR = ROOT / CFG["paths"]["models_dir"]
ckpts = sorted(MODEL_DIR.glob("*.pth"))
if not ckpts:
    log.error("No checkpoints in %s", MODEL_DIR); sys.exit(1)

arch2models: Dict[str, List[torch.nn.Module]] = {}
for ck in ckpts:
    arch = "efficientnet_b0" if "efficientnet" in ck.name else "regnety_800mf"
    mdl = init_model(arch)
    mdl.load_state_dict(torch.load(ck, map_location="cpu")["model_state_dict"],
                        strict=False)
    mdl.to(device).eval()
    arch2models.setdefault(arch, []).append(mdl)
    log.info("Loaded %s", ck.name)

# sanity check – need both architectures
if not all(a in arch2models for a in ("efficientnet_b0", "regnety_800mf")):
    log.error("Need checkpoints for both EfficientNet-B0 and RegNetY-800MF"); sys.exit(1)

# ─────────────── update loop ────────────────
updated = 0
with torch.no_grad():
    for X, paths in loader:
        X = X.to(device, non_blocking=True)

        # HARD ensemble inside each architecture (element-wise min over runs)
        probs_arch: Dict[str, torch.Tensor] = {}
        for arch, models in arch2models.items():
            runs = torch.stack([torch.softmax(m(X), 1) for m in models])  # R×B×C
            probs_arch[arch] = runs.min(0).values                           # B×C

        ensemble = 0.5*(probs_arch["efficientnet_b0"] + probs_arch["regnety_800mf"])
        ensemble = ensemble.cpu().numpy()  # B×C

        for p_new, label_path in zip(ensemble, paths):
            lp = Path(label_path)
            if not lp.exists():
                log.warning("Missing %s", lp); continue
            p_old = np.load(lp).astype(np.float32)
            p_blend = (1.0 - K)*p_old + K*p_new
            if not args.dry_run:
                np.save(lp, p_blend)
            updated += 1

log.info("%sUpdated %d label vectors.",
         "[dry-run] " if args.dry_run else "", updated)

#!/usr/bin/env python
"""
regnety.py – BirdCLEF 2025 trainer (fixed)
========================================
Trains **CFG.REG_NUM_MODELS** class‑conditional RegNetY‑0.8GF models on the
pre‑processed mel‑spectrogram chunks.  This revision fixes the CSV‑loading
bugs flagged in the review:

* **Uses `pd.read_csv` / `pd.concat`** instead of the erroneous
  `Path.read_csv()` call.
* Adds the missing **`import pandas as pd`** statement.
* Keeps all earlier improvements (model‑name guard, CUDA pin‑memory toggle,
  checkpoint arch string, seed offset, CLI alias, etc.).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd  # <‑‑‑ FIXED: explicit pandas import
import torch
import torch.nn.functional as F
import timm
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from configure import CFG
from data_utils import FileWiseSampler, MelDataset, seed_everything

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _get_model_name() -> str:
    for name in ("regnety_008", "regnety_008gf"):
        if name in timm.list_models():
            return name
    raise RuntimeError("RegNetY‑0.8GF backbone missing in timm build.")


def _soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.dtype == torch.long:
        return F.cross_entropy(logits, targets, reduction="none")
    return -(targets * torch.log_softmax(logits, dim=1)).sum(1)


def _logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("regnety")

# ────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────

class _Trainer:
    def __init__(self, rid: int, n_cls: int, arch: str, log: logging.Logger):
        self.rid, self.log = rid, log
        self.device = torch.device("cuda" if CFG.use_cuda() else "cpu")
        self.model = timm.create_model(arch, pretrained=True, in_chans=1,
                                       num_classes=n_cls).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=CFG.REG_LR,
                               weight_decay=CFG.REG_WEIGHT_DECAY)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=CFG.REG_EPOCHS)
        self.scaler = GradScaler(enabled=CFG.use_cuda())

    def _step(self, batch):
        x, y, w = (t.to(self.device, non_blocking=True) for t in batch)
        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.use_cuda()):
            loss = (_soft_ce(self.model(x), y) * w).mean()
        self.scaler.scale(loss).backward(); self.scaler.step(self.opt); self.scaler.update()
        return float(loss.detach())

    def fit(self, loader: DataLoader):
        for ep in range(1, CFG.REG_EPOCHS + 1):
            self.model.train(); run = 0.0
            for batch in loader:
                run += self._step(batch) * batch[0].size(0)
            self.sched.step()
            self.log.info("run=%d ep=%d/%d loss=%.5f lr=%.2e", self.rid, ep,
                          CFG.REG_EPOCHS, run / len(loader.dataset), self.sched.get_last_lr()[0])

    def save(self, out: Path, arch: str, s2i: Dict[str, int]):
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": arch, "model": self.model.state_dict(), "species2idx": s2i},
                   out / f"{arch}_run{self.rid}.pth")
        self.log.info("✔ saved checkpoint run%d", self.rid)

# ────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────

def _load_meta() -> pd.DataFrame:  # <‑‑‑ FIXED: correct pandas usage
    df = pd.read_csv(CFG.PROCESSED_DIR / "train_metadata.csv")
    sc = CFG.PROCESSED_DIR / "soundscape_metadata.csv"
    if sc.exists():
        df = pd.concat([df, pd.read_csv(sc)], ignore_index=True)
    return df

# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    cli = argparse.ArgumentParser(description="Train RegNetY‑0.8GF ensemble")
    cli.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto")
    args = cli.parse_args()
    if args.device != "auto":
        CFG.DEVICE = args.device  # type: ignore[attr-defined]

    seed_everything(CFG.SEED)
    log = _logger()
    log.info("Device: %s", CFG.DEVICE)

    df = _load_meta()
    if df.empty:
        log.error("No metadata – run process.py first."); return

    cls_set = {k for js in df["label_json"] for k in json.loads(js)}
    classes = sorted(cls_set); s2i = {s: i for i, s in enumerate(classes)}
    log.info("Classes: %d", len(classes))

    data = MelDataset(df, s2i, augment=True)
    df["_src"] = df["mel_path"].str.extract(r"([^/]+)_\d+s.npy$", expand=False)
    loader = DataLoader(data, batch_size=CFG.REG_BATCH_SIZE,
                        sampler=FileWiseSampler(df, "_src"),
                        num_workers=CFG.REG_NUM_WORKERS,
                        pin_memory=CFG.use_cuda(), drop_last=True)

    arch = _get_model_name()
    for run in range(1, CFG.REG_NUM_MODELS + 1):
        seed_everything(CFG.SEED + 1000 + run)
        t = _Trainer(run, len(classes), arch, log); t.fit(loader); t.save(CFG.REG_MODEL_DIR, arch, s2i)
        del t; torch.cuda.empty_cache()

    log.info("🏁 Finished – checkpoints in %s", CFG.REG_MODEL_DIR)


if __name__ == "__main__":
    main()

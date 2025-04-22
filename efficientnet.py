#!/usr/bin/env python3
"""
efficientnet.py – EfficientNet‑B0 ensemble trainer for **BirdCLEF 2025**
=====================================================================
Trains *CFG.EFF_NUM_MODELS* EfficientNet‑B0 classifiers on the mel‑spectrogram
chunks produced by ``process.py``.  The script follows the 2024 winning recipe
and the workflow requested by the user:

* Cross‑entropy with **soft‑label** support
* One‑chunk‑per‑file sampling via ``FileWiseSampler``
* SpecAugment + CutMix enabled through *data_utils*
* Cosine‑annealing LR schedule & mixed‑precision (AMP)
* Optional device override via ``--device`` (auto / cpu / gpu)
* Reproducible ensemble – each run uses a different seed offset
* Checkpoints saved under ``CFG.EFF_MODEL_DIR`` as
  ``efficientnet_b0_run{RUN}.pth`` (state + species mapping)

Usage
-----
```bash
# train two models on GPU (default)
python efficientnet.py              #   → models/efficientnet/*.pth

# force CPU training & 5 epochs only (quick test)
python efficientnet.py --device cpu --epochs 5
```
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from configure import CFG
from data_utils import FileWiseSampler, MelDataset, seed_everything

# ───────────────────────── logging ──────────────────────────

def _logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("efficientnet")


# ──────────────────── soft‑label CE helper ───────────────────

def _soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross‑entropy that supports *either* hard indices *or* soft vectors."""
    if targets.dtype == torch.long:
        return F.cross_entropy(logits, targets, reduction="none")
    logp = torch.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1)


# ───────────────────────── trainer ───────────────────────────

class _Trainer:
    """Encapsulates one EfficientNet run (different seed / init)."""

    def __init__(self, run_id: int, classes: List[str], log: logging.Logger):
        self.run_id, self.log = run_id, log
        self.device = torch.device("cuda" if CFG.use_cuda() else "cpu")
        self.classes = classes

        # model – EfficientNet‑B0  (1‑channel input)
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=len(classes),
        ).to(self.device)

        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=CFG.EFF_LR,
            weight_decay=CFG.EFF_WEIGHT_DECAY,
        )
        self.sched = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=CFG.EFF_EPOCHS
        )
        self.scaler = GradScaler(enabled=CFG.use_cuda())

    # ‑‑ one optimisation step ‑─────────────────────────────
    def _step(self, batch):
        x, y, w = (t.to(self.device, non_blocking=True) for t in batch)
        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=CFG.use_cuda()):
            logits = self.model(x)
            loss = (_soft_ce(logits, y) * w).mean()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.detach())

    # ‑‑ epoch loop ‑─────────────────────────────────────────
    def fit(self, loader: DataLoader) -> None:
        for ep in range(1, self._epochs + 1):
            self.model.train(); run_loss = 0.0
            for batch in loader:
                run_loss += self._step(batch) * batch[0].size(0)
            self.sched.step()
            self.log.info(
                "run=%d  epoch=%d/%d  loss=%.5f  lr=%.2e",
                self.run_id,
                ep,
                self._epochs,
                run_loss / len(loader.dataset),
                self.sched.get_last_lr()[0],
            )

    # property wrapper to cope with CLI‑overridden epochs
    @property
    def _epochs(self) -> int:
        return getattr(CFG, "EFF_EPOCHS", 10)

    # ‑‑ save checkpoint ‑────────────────────────────────────
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "arch": "efficientnet_b0",
            "model": self.model.state_dict(),
            "species2idx": {s: i for i, s in enumerate(self.classes)},
        }
        fp = out_dir / f"efficientnet_b0_run{self.run_id}.pth"
        torch.save(ckpt, fp)
        self.log.info("✔ saved checkpoint → %s", fp.name)


# ─────────────────── metadata & DataLoader ───────────────────

def _load_meta() -> pd.DataFrame:
    df = pd.read_csv(CFG.PROCESSED_DIR / "train_metadata.csv")
    sc = CFG.PROCESSED_DIR / "soundscape_metadata.csv"
    if sc.exists():
        df = pd.concat([df, pd.read_csv(sc)], ignore_index=True)
    return df


# ─────────────────────────── main ────────────────────────────

def main() -> None:  # noqa: D401
    cli = argparse.ArgumentParser(description="Train EfficientNet‑B0 ensemble")
    cli.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto")
    cli.add_argument("--epochs", type=int, default=None, help="override CFG.EFF_EPOCHS")
    args = cli.parse_args()

    # runtime config overrides – device & epochs
    if args.device != "auto":
        CFG.DEVICE = args.device  # type: ignore[attr-defined]
    if args.epochs is not None:
        CFG.EFF_EPOCHS = args.epochs  # type: ignore[attr-defined]

    seed_everything(CFG.SEED)
    log = _logger()
    log.info("Device: %s", CFG.DEVICE)

    df = _load_meta()
    if df.empty:
        log.error("No metadata found – run process.py first."); return

    # Determine full class list from metadata (robust to pruning)
    cls_set = {k for js in df["label_json"] for k in json.loads(js)}
    classes = sorted(cls_set)
    s2i = {s: i for i, s in enumerate(classes)}
    log.info("Classes: %d", len(classes))

    # DataLoader – one random 10‑s chunk per source file each epoch
    dataset = MelDataset(df, s2i, augment=True)
    df["_src"] = df["mel_path"].str.extract(r"([^/]+)_\d+s.npy$", expand=False)
    loader = DataLoader(
        dataset,
        batch_size=CFG.EFF_BATCH_SIZE,
        sampler=FileWiseSampler(df, "_src"),
        num_workers=CFG.EFF_NUM_WORKERS,
        pin_memory=CFG.use_cuda(),
        drop_last=True,
    )

    # Ensemble training loop -------------------------------------------------
    for run in range(1, CFG.EFF_NUM_MODELS + 1):
        seed_everything(CFG.SEED + run)  # new seed each run
        trainer = _Trainer(run, classes, log)
        trainer.fit(loader)
        trainer.save(CFG.EFF_MODEL_DIR)
        del trainer; torch.cuda.empty_cache()

    log.info("🏁 Finished – checkpoints in %s", CFG.EFF_MODEL_DIR)


if __name__ == "__main__":
    main()

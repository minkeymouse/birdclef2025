#!/usr/bin/env python3
"""
dataloader.py – dataset, sampler, and training loop utilities
============================================================
Shared by train_efficientnet.py and train_regnety.py.

Classes:
- BirdClefDataset   — loads mel-spectrogram chunks and soft-label vectors from metadata.
- create_dataloader — builds a PyTorch DataLoader with optional WeightedRandomSampler.
- train_model      — generic training loop with CE or Focal loss and sample weighting,
                     saving best checkpoints by loss (imported from metrics module).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# project imports
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root))
from src.utils.metrics import macro_auc_score, macro_precision_score
from src.train.loss import get_criterion

__all__ = ["BirdClefDataset", "create_dataloader", "train_model"]

# ──────────────────────────────────────────────────────────────────────────────
# Choose your loss here: "cross_entropy" or "focal_loss_bce"
LOSS_TYPE = "cross_entropy"
# If using focal_loss_bce, you can also adjust these:
FOCAL_ALPHA    = 0.25
FOCAL_GAMMA    = 2
BCE_WEIGHT     = 0.6
FOCAL_WEIGHT   = 1.4
REDUCTION_MODE = "mean"
# ──────────────────────────────────────────────────────────────────────────────

class BirdClefDataset(Dataset):
    """Dataset for BirdCLEF: loads mel-spectrograms and label vectors from metadata."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        class_map: Dict[str, int],
        *,
        mel_shape: Optional[Tuple[int, int]] = None,
        augment: bool = False,
    ) -> None:
        self.df = metadata_df.reset_index(drop=True)
        self.class_map = class_map
        self.num_classes = len(class_map)
        self.augment = augment

        # infer mel_shape if not provided
        if mel_shape is None:
            sample_path = self.df.iloc[0]["mel_path"]
            sample = np.load(sample_path)
            self.mel_shape = sample.shape
        else:
            self.mel_shape = mel_shape

        # extract sample weights
        self._weights = self.df.get("weight", pd.Series(1.0, index=self.df.index))
        self._weights = self._weights.fillna(1.0).astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        mel_path = row.get("mel_path")
        label_path = row.get("label_path")
        if not mel_path or pd.isna(mel_path):
            raise FileNotFoundError("Missing 'mel_path' in metadata.")
        if not label_path or pd.isna(label_path):
            raise FileNotFoundError("Missing 'label_path' in metadata.")

        mel = np.load(mel_path)
        label_vec = np.load(label_path).astype(np.float32)

        # resize if shape mismatch
        if mel.shape != tuple(self.mel_shape):
            mel = cv2.resize(mel, (self.mel_shape[1], self.mel_shape[0]))
        if self.augment:
            mel = self._augment_mel(mel)

        # convert to 3-channel image tensor
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1).float()
        weight = torch.tensor(self._weights[index], dtype=torch.float32)
        return mel_tensor, torch.from_numpy(label_vec), weight

    def _augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Random time-frequency masking (SpecAugment-style)."""
        h, w = mel.shape
        # freq mask
        for _ in range(np.random.randint(1, 3)):
            f0 = np.random.randint(0, h)
            f_len = np.random.randint(5, max(6, h // 8 + 1))
            mel[f0 : f0 + f_len, :] = mel.mean()
        # time mask
        for _ in range(np.random.randint(1, 3)):
            t0 = np.random.randint(0, w)
            t_len = np.random.randint(5, max(6, w // 8 + 1))
            mel[:, t0 : t0 + t_len] = mel.mean()
        return mel


def create_dataloader(
    dataset: BirdClefDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Build DataLoader; use WeightedRandomSampler if sample weights vary."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4

    weights = getattr(dataset, "_weights", None)
    if weights is not None and not np.allclose(weights, 1.0):
        sampler = WeightedRandomSampler(
            weights=list(weights),
            num_samples=len(dataset),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> List[str]:
    """
    Train the model, save the best checkpoint (by validation loss), and return its path.
    Loss function is chosen via the module‐level LOSS_TYPE constant.
    """
    # ———————— optimizer & scheduler setup ————————
    epochs   = int(config["training"]["epochs"])
    opt_cfg  = config.get("optimizer", {})
    lr       = float(opt_cfg.get("lr", 1e-3))
    wd       = float(opt_cfg.get("weight_decay", 0.0))
    opt_type = opt_cfg.get("type", "adamw").lower()

    optimizer = (
        optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt_type == "adamw"
        else optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    )

    sch_cfg   = config.get("scheduler", {})
    scheduler = None
    if sch_cfg.get("type") == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sch_cfg.get("T_max", epochs)),
            eta_min=float(sch_cfg.get("eta_min", 1e-6)),
        )

    # ———————— instantiate loss criterion ————————
    criterion = get_criterion(
        loss_type=LOSS_TYPE,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        reduction=REDUCTION_MODE,
        bce_weight=BCE_WEIGHT,
        focal_weight=FOCAL_WEIGHT,
    )

    # ———————— training loop ————————
    best_loss: float       = float("inf")
    best_ckpt: Optional[str] = None
    mdir = Path(config["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    arch   = config.get("current_arch", "model")
    run_id = config.get("current_run", 1)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0

        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            raw_loss = criterion(model(x), y)
            loss     = (raw_loss * w).mean()
            loss.backward()
            optimizer.step()
            total_train += loss.item() * x.size(0)

        if scheduler:
            scheduler.step()
        avg_train_loss = total_train / len(train_loader.dataset)

        # ———————— validation loop ————————
        model.eval()
        total_val, preds, gts = 0.0, [], []

        with torch.no_grad():
            for x, y, w in val_loader:
                x, y, w    = x.to(device), y.to(device), w.to(device)
                logits     = model(x)
                raw_val    = criterion(logits, y)
                total_val += (raw_val * w).mean().item() * x.size(0)
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
                gts.append(y.cpu().numpy())

        avg_val_loss = total_val / len(val_loader.dataset)
        y_pred       = np.vstack(preds)
        y_true       = np.vstack(gts)

        # ———————— metrics & checkpointing ————————
        val_auc  = macro_auc_score(y_true, y_pred)
        val_prec = macro_precision_score(y_true, y_pred)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f} | "
            f"AUC {val_auc:.4f} | Precision {val_prec:.4f}"
        )

        ckpt_name = (
            f"{arch}_run{run_id}_epoch{epoch}"
            f"_loss{avg_val_loss:.4f}_{int(time.time())}.pth"
        )
        ckpt_path = mdir / ckpt_name

        if avg_val_loss < best_loss - 1e-6:
            if best_ckpt:
                try:
                    os.remove(best_ckpt)
                except OSError:
                    pass
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "class_map": config.get("class_map")},
                str(ckpt_path),
            )
            best_loss = avg_val_loss
            best_ckpt = str(ckpt_path)
            print(f"  → new best checkpoint: {ckpt_path} (val loss {avg_val_loss:.4f})")

    return [best_ckpt] if best_ckpt else []


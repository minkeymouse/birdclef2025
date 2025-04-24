#!/usr/bin/env python3
"""
dataloader.py – dataset, sampler, and training loop utilities
============================================================
Shared by *train_efficientnet.py* and *train_regnety.py*.

* **BirdClefDataset**   — loads 10‑second mel‑spectrogram *npy* files + soft‑label
  vectors (either from ``label_path`` or inline JSON columns).
* **create_dataloader** — builds a PyTorch DataLoader with optional
  ``WeightedRandomSampler`` so minority species can be up‑sampled.
* **train_model**       — generic training loop using **Soft Cross‑Entropy**
  (multi‑class, soft‑label) with per‑sample weights and top‑3 checkpointing
  by macro‑AUC.
* **macro_auc_score / macro_precision_score** — simple sklearn‑powered metrics
  that gracefully skip empty classes.

The implementation follows the high‑level rules you outlined:
• Primary label = 1.0, secondary labels = 0.05 (already encoded in metadata)
• Cross‑Entropy (soft) instead of BCE
• Checkpoints saved in ``models/<arch>/`` with epoch & timestamp
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, roc_auc_score
from torch import nn, optim
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              default_collate)

__all__ = [
    "BirdClefDataset",
    "create_dataloader",
    "train_model",
]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class BirdClefDataset(Dataset):
    """Load mel‑spectrogram chunks + soft labels from *train_metadata.csv*."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        class_map: Dict[str, int],
        *,
        mel_shape: Tuple[int, int] = (128, 256),  # (mel_bins, time_frames)
        augment: bool = False,
    ) -> None:
        self.df = metadata_df.reset_index(drop=True)
        self.class_map = class_map
        self.num_classes = len(class_map)
        self.mel_shape = mel_shape
        self.augment = augment

        # -------- pre‑extract labels + sample weights for speed --------------
        self._labels: np.ndarray = np.zeros((len(self.df), self.num_classes),
                                            dtype=np.float32)
        self._weights: np.ndarray = np.ones(len(self.df), dtype=np.float32)

        for i, row in self.df.iterrows():
            # 1) Prefer vector stored in separate .npy file
            if "label_path" in row and pd.notna(row["label_path"]):
                self._labels[i] = np.load(row["label_path"]).astype(np.float32)
            # 2) JSON string column ("{"sp1":1.0,"sp2":0.05,...}")
            elif "label_json" in row and isinstance(row["label_json"], str):
                js = json.loads(row["label_json"])
                for sp, v in js.items():
                    if sp in self.class_map:
                        self._labels[i, self.class_map[sp]] = float(v)
            # 3) Fallback one‑hot from primary / secondary columns
            else:
                primary = row.get("primary_label")
                if pd.notna(primary) and primary in self.class_map:
                    self._labels[i, self.class_map[primary]] = 1.0
                sec_raw = row.get("secondary_labels", "")
                if isinstance(sec_raw, str) and sec_raw.strip():
                    for sp in sec_raw.split():
                        if sp in self.class_map:
                            self._labels[i, self.class_map[sp]] = 0.05
            # weight column (default 1.0)
            self._weights[i] = float(row.get("weight", 1.0))

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 – simple length
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ---- load / resize mel ------------------------------------------------
        mel: np.ndarray
        if "mel_path" in row and pd.notna(row["mel_path"]):
            mel_path = row["mel_path"]
            mel = np.load(mel_path)
        else:
            raise FileNotFoundError("Metadata must contain 'mel_path' column.")

        if mel.shape != tuple(self.mel_shape):
            mel = cv2.resize(mel, (self.mel_shape[1], self.mel_shape[0]))

        if self.augment:
            mel = self._augment_mel(mel)

        mel_tensor = torch.from_numpy(mel).float()
        mel_tensor = mel_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

        label_vec = torch.from_numpy(self._labels[idx])
        sample_w = torch.tensor(self._weights[idx], dtype=torch.float32)
        return mel_tensor, label_vec, sample_w

    # ------------------------------------------------------------------
    def _augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Random time‑freq masking (SpecAugment‑style)."""
        h, w = mel.shape
        for _ in range(np.random.randint(1, 3)):
            f0, f_len = np.random.randint(0, h), np.random.randint(5, h // 8 + 1)
            mel[f0 : f0 + f_len] = mel.mean()
        for _ in range(np.random.randint(1, 3)):
            t0, t_len = np.random.randint(0, w), np.random.randint(5, w // 8 + 1)
            mel[:, t0 : t0 + t_len] = mel.mean()
        return mel


# -----------------------------------------------------------------------------
# DataLoader helper
# -----------------------------------------------------------------------------

def create_dataloader(
    dataset: BirdClefDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Return a DataLoader with optional *WeightedRandomSampler*."""

    if hasattr(dataset, "_weights") and not np.allclose(dataset._weights, 1.0):
        sampler = WeightedRandomSampler(dataset._weights, len(dataset), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0 if os.name == "nt" else 4,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if os.name == "nt" else 4,
        pin_memory=True,
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan  # single‑class or constant case


def macro_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro‑average ROC‑AUC; skip classes with no positives."""
    n_classes = y_true.shape[1]
    aucs = [
        _safe_auc(y_true[:, c], y_pred[:, c]) for c in range(n_classes)
    ]
    aucs = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else 0.0


def macro_precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5
) -> float:
    """Macro‑averaged precision at given threshold."""
    y_bin = (y_pred >= threshold).astype(int)
    precisions: List[float] = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:  # skip no‑positive class
            continue
        precisions.append(
            precision_score(y_true[:, c], y_bin[:, c], zero_division=0)
        )
    return float(np.mean(precisions)) if precisions else 0.0


# -----------------------------------------------------------------------------
# Training loop (soft‑label Cross‑Entropy)
# -----------------------------------------------------------------------------

def _soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft Cross‑Entropy for probability targets on *multi‑class* outputs."""
    log_prob = torch.log_softmax(logits, dim=1)
    return -(targets * log_prob).sum(dim=1)  # per‑sample loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Generic supervised trainer saving top‑3 checkpoints by macro‑AUC."""

    epochs = int(config["training"]["epochs"])
    opt_cfg = config["optimizer"]
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.0))

    optimizer = (
        optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt_cfg["type"].lower() == "adamw"
        else optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    )

    # Scheduler ------------------------------------------------------------
    scheduler = None
    sch_cfg = config.get("scheduler")
    if sch_cfg and sch_cfg["type"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sch_cfg.get("T_max", epochs)),
            eta_min=float(sch_cfg.get("eta_min", 1e-6)),
        )

    best_scores: List[float] = []
    best_ckpts: List[str] = []

    for epoch in range(1, epochs + 1):
        # ---------------------------- train ------------------------------
        model.train()
        total_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = (_soft_ce_loss(logits, y) * w).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        if scheduler:
            scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)

        # --------------------------- validate ----------------------------
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = model(x.to(device))
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
                gts.append(y.numpy())
        y_pred = np.vstack(preds)
        y_true = np.vstack(gts)

        val_auc = macro_auc_score(y_true, y_pred)
        val_prec = macro_precision_score(y_true, y_pred)
        print(
            f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | AUC {val_auc:.4f} | P@0.5 {val_prec:.4f}"
        )

        # ------------------------ checkpointing --------------------------
        mdir = Path(config["paths"]["models_dir"])
        mdir.mkdir(parents=True, exist_ok=True)
        arch = config["current_arch"]
        run = config.get("current_run", 1)
        fname = (
            f"{arch}_run{run}_epoch{epoch}_auc{val_auc:.4f}_{int(time.time())}.pth"
        )
        fpath = str(mdir / fname)

        if len(best_scores) < 3 or val_auc > min(best_scores):
            torch.save({"model_state_dict": model.state_dict(), "class_map": config["class_map"]}, fpath)
            if len(best_scores) < 3:
                best_scores.append(val_auc)
                best_ckpts.append(fpath)
            else:
                worst = int(np.argmin(best_scores))
                try:
                    os.remove(best_ckpts[worst])
                except FileNotFoundError:
                    pass
                best_scores[worst] = val_auc
                best_ckpts[worst] = fpath
            print("  → checkpoint saved", fpath)

    # sort best checkpoints best‑to‑worst before returning
    order = np.argsort(best_scores)[::-1]
    return [best_ckpts[i] for i in order]

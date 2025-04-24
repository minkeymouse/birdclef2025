#!/usr/bin/env python3
"""
dataloader.py – dataset, sampler, and training loop utilities
============================================================
Shared by train_efficientnet.py and train_regnety.py.

Classes:
- BirdClefDataset   — loads mel-spectrogram chunks and soft-label vectors from metadata.
- create_dataloader — builds a PyTorch DataLoader with optional WeightedRandomSampler.
- train_model      — generic training loop with Soft Cross-Entropy and sample weighting,
                     saving top-3 checkpoints by macro-AUC (imported from metrics module).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import sys

project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils.metrics import macro_auc_score, macro_precision_score

__all__ = ["BirdClefDataset", "create_dataloader", "train_model"]


class BirdClefDataset(Dataset):
    """Dataset for BirdCLEF: loads mel-spectrograms and soft-label vectors."""

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

        # Pre-extract labels and weights for speed
        self._labels = np.zeros((len(self.df), self.num_classes), dtype=np.float32)
        self._weights = np.ones(len(self.df), dtype=np.float32)
        for idx, row in self.df.iterrows():
            self._labels[idx] = self._extract_labels(row)
            self._weights[idx] = float(row.get("weight", 1.0))

    def _extract_labels(self, row: pd.Series) -> np.ndarray:
        vec = np.zeros(self.num_classes, dtype=np.float32)
        # 1) Load vector from .npy file
        if row.get("label_path") and pd.notna(row["label_path"]):
            vec = np.load(row["label_path"]).astype(np.float32)
        # 2) JSON string column
        elif row.get("label_json") and isinstance(row["label_json"], str):
            js = json.loads(row["label_json"])
            for sp, v in js.items():
                if sp in self.class_map:
                    vec[self.class_map[sp]] = float(v)
        # 3) Fallback: primary and secondary labels
        else:
            primary = row.get("primary_label")
            if pd.notna(primary) and primary in self.class_map:
                vec[self.class_map[primary]] = 1.0
            sec_raw = row.get("secondary_labels", "")
            if isinstance(sec_raw, str) and sec_raw.strip():
                for sp in sec_raw.split():
                    if sp in self.class_map:
                        vec[self.class_map[sp]] = 0.05
        return vec

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        mel_path = row.get("mel_path")
        if not mel_path or pd.isna(mel_path):
            raise FileNotFoundError("Missing 'mel_path' in metadata.")
        mel = np.load(mel_path)
        if mel.shape != tuple(self.mel_shape):
            mel = cv2.resize(mel, (self.mel_shape[1], self.mel_shape[0]))
        if self.augment:
            mel = self._augment_mel(mel)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1).float()
        label_vec = torch.from_numpy(self._labels[index])
        weight = torch.tensor(self._weights[index], dtype=torch.float32)
        return mel_tensor, label_vec, weight

    def _augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Random time-frequency masking (SpecAugment-style)."""
        h, w = mel.shape
        for _ in range(np.random.randint(1, 3)):
            f0 = np.random.randint(0, h)
            f_len = np.random.randint(5, h // 8 + 1)
            mel[f0 : f0 + f_len, :] = mel.mean()
        for _ in range(np.random.randint(1, 3)):
            t0 = np.random.randint(0, w)
            t_len = np.random.randint(5, w // 8 + 1)
            mel[:, t0 : t0 + t_len] = mel.mean()
        return mel


def create_dataloader(
    dataset: BirdClefDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Build DataLoader; use WeightedRandomSampler if sample weights vary."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4
    if hasattr(dataset, "_weights") and not np.allclose(dataset._weights, 1.0):
        sampler = WeightedRandomSampler(
            weights=list(dataset._weights),
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


def _soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft Cross-Entropy loss for probability targets."""
    log_prob = torch.log_softmax(logits, dim=1)
    return -(targets * log_prob).sum(dim=1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> List[str]:
    """Train model, save top-3 checkpoints by macro-AUC, and return their paths."""
    epochs = int(config["training"]["epochs"])
    opt_cfg = config["optimizer"]
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    opt_type = opt_cfg.get("type", "adamw").lower()
    optimizer = (
        optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt_type == "adamw"
        else optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    )

    scheduler = None
    sch_cfg = config.get("scheduler", {})
    if sch_cfg.get("type") == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sch_cfg.get("T_max", epochs)),
            eta_min=float(sch_cfg.get("eta_min", 1e-6)),
        )

    best_scores: List[float] = []
    best_ckpts: List[str] = []
    mdir = Path(config["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    arch = config.get("current_arch", "model")
    run_id = config.get("current_run", 1)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            loss = (_soft_ce_loss(model(x), y) * w).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        if scheduler:
            scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)

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
            f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | "
            f"AUC {val_auc:.4f} | Precision {val_prec:.4f}"
        )

        # Checkpointing
        fname = f"{arch}_run{run_id}_epoch{epoch}_auc{val_auc:.4f}_{int(time.time())}.pth"
        fpath = str(mdir / fname)
        if len(best_scores) < 3 or val_auc > min(best_scores):
            torch.save({"model_state_dict": model.state_dict(), "class_map": config.get("class_map")}, fpath)
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
            print(f"  → checkpoint saved: {fpath}")

    # Return sorted checkpoints by descending AUC
    order = sorted(range(len(best_scores)), key=lambda i: best_scores[i], reverse=True)
    return [best_ckpts[i] for i in order]

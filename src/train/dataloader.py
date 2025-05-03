#!/usr/bin/env python3
"""
dataloader.py – dataset, sampler, and training loop utilities
============================================================
Shared by train_efficientnet.py and train_regnety.py.

Classes:
- BirdClefDataset   — loads mel-spectrogram chunks and soft-label vectors from metadata.
- collatefn
- create_dataloader — builds a PyTorch DataLoader with optional WeightedRandomSampler.
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
import random

__all__ = ["BirdClefDataset", "create_dataloader", "train_model"]

class BirdClefDataset(Dataset):
    """Dataset for BirdCLEF: loads mel-spectrograms and label vectors from metadata."""

    def __init__(
        self,
        label2id,
        metadata_df,
        num_classes,
        *,
        mode: str = "train"
    ) -> None:
        self.df = metadata_df.reset_index(drop=True)
        self.sample_weights = self.df["weight"].astype(float).tolist()
        self.mel_shape = (256, 256)
        self.label2id = label2id
        self.num_classes = num_classes
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])
        label = np.load(row["label_path"]).astype(np.float32)
        if self.mode == "train":
            mel = self.apply_spec_augmentations(mel)

        return {
            "melspec": torch.tensor(mel, dtype=torch.float32),
            "label":    torch.tensor(label, dtype=torch.float32),
            "weight":   torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "filename": row["filename"],
        }
    
    def apply_spec_augmentations(self, spec):
        """Apply augmentations to spectrogram"""
        # Time masking (horizontal stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0
        
        # Frequency masking (vertical stripes)
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0
        
        # Random brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1) 
            
        return spec

    def _weights(self):
        return self.sample_weights

def collate_fn(batch):
    batch = [b for b in batch if b]
    keys  = batch[0].keys()
    result = {k: [] for k in keys}
    for b in batch:
        for k, v in b.items():
            result[k].append(v)
    # Stack tensors
    for k in ["mel", "label", "weight"]:
        if k in result:
            result[k] = torch.stack(result[k])
    return result

def create_dataloader(dataset, batch_size, num_workers=None, pin_memory=True):
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4
    sampler = WeightedRandomSampler(
        weights=dataset._weights(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


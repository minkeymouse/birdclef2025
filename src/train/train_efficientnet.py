#!/usr/bin/env python3
"""
train_efficientnet.py – EfficientNet-B0 ensemble trainer
======================================================
Train **N** EfficientNet-B0 models (different random seeds) on the BirdCLEF
10-second-chunk dataset using a YAML configuration file.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import timm

import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

# project imports
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "train.yaml"
import sys
sys.path.insert(0, str(project_root))
from src.train.dataloader import BirdClefDataset, create_dataloader, train_model
from src.utils import utils

with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

dataset_cfg = CFG["dataset"]
model_cfg = CFG["model"]
training_cfg = CFG["training"]
optimizer_cfg = CFG["selection"]
loss_cfg = CFG["loss"]
scheduler_cfg = CFG["scheduler"]
paths_cfg = CFG["paths"]

def main() -> None:

    # Identify EfficientNet block
    arch_cfg = next(
        (a for a in model_cfg["architecture"] if a["name"].startswith("efficientnet")),
        None,
    )
    if arch_cfg is None:
        raise ValueError("EfficientNet config block missing in configuration file.")
    model_name = arch_cfg["name"]
    num_models = arch_cfg.get("num_models", 1)

    # ----------------------------------
    # Load and prepare metadata
    # ----------------------------------
    df_meta = pd.read_csv(dataset_cfg["train_metadata"])

    class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), dataset_cfg["train_csv"])

    # Train/validation split
    val_frac = float(training_cfg["training"].get("val_fraction", 0.1))
    seed = int(training_cfg["training"].get("seed", 42))
    df_val = df_meta.sample(frac=val_frac, random_state=seed)
    df_train = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # ----------------------------------
    # Datasets & DataLoaders
    # ----------------------------------
    mel_shape = tuple(dataset_cfg.get("target_shape", [256, 256]))
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 4))

    train_ds = BirdClefDataset(df_train, class_map, mel_shape=mel_shape, augment=True)
    val_ds = BirdClefDataset(df_val, class_map, mel_shape=mel_shape, augment=False)

    train_loader = create_dataloader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = create_dataloader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ----------------------------------
    # Device setup
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ----------------------------------
    # Checkpoint directory
    # ----------------------------------
    ckpt_root = Path(paths_cfg.get("models_dir", "models"))
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # Training runs
    # ----------------------------------
    saved_ckpts: list[str] = []
    base_seed = seed

    for run in range(1, num_models + 1):
        torch.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)

        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Linear(in_feats, len(class_map))
        model.to(device)

        # Set context for train_model
        CFG["current_arch"] = model_name
        CFG["current_run"] = run
        CFG["class_map"] = class_map

        # Train and collect checkpoints
        ckpts = train_model(model, train_loader, val_loader, CFG, device)
        saved_ckpts.extend(ckpts)

    # ----------------------------------
    # Summary
    # ----------------------------------
    print("\nTraining complete – saved checkpoints:")
    for p in saved_ckpts:
        print(" •", p)


if __name__ == "__main__":
    main()
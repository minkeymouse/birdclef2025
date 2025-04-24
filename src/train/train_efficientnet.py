#!/usr/bin/env python3
"""
train_efficientnet.py – EfficientNet-B0 ensemble trainer
======================================================
Train **N** EfficientNet-B0 models (different random seeds) on the BirdCLEF
10-second-chunk dataset using a YAML configuration file.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

# project imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.train.dataloader import BirdClefDataset, create_dataloader, train_model
from src.utils import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ensemble of EfficientNet-B0 models on BirdCLEF data"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=project_root / "config" / "train.yaml",
        help="Path to training config YAML"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    # Pull sections
    dataset_cfg   = CFG["dataset"]
    model_cfg     = CFG["model"]
    training_cfg  = CFG["training"]
    optimizer_cfg = CFG.get("optimizer", {})
    loss_cfg      = CFG.get("loss", {})
    scheduler_cfg = CFG.get("scheduler", {})
    paths_cfg     = CFG.get("paths", {})

    # ----------------------------------
    # Identify EfficientNet config
    # ----------------------------------
    archs = model_cfg.get("architectures", [])
    eff_cfg = next(
        (a for a in archs if a.get("name", "").lower().startswith("efficientnet")),
        None,
    )
    if eff_cfg is None:
        raise ValueError("No EfficientNet entry found under model.architectures in config.")
    model_name = eff_cfg["name"]
    num_models = int(eff_cfg.get("num_models", 1))

    # ----------------------------------
    # Load and prepare metadata
    # ----------------------------------
    df_meta = pd.read_csv(dataset_cfg["train_metadata"] )
    class_list, class_map = utils.load_taxonomy(
        paths_cfg.get("taxonomy_csv"),
        dataset_cfg.get("train_csv")
    )

    # Train/validation split
    val_frac = float(training_cfg.get("val_fraction", 0.1))
    base_seed = int(training_cfg.get("seed", 42))
    df_val   = df_meta.sample(frac=val_frac, random_state=base_seed)
    df_train = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    # ----------------------------------
    # Datasets & DataLoaders
    # ----------------------------------
    mel_shape   = tuple(dataset_cfg.get("target_shape", [256, 256]))
    batch_size  = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))

    train_ds = BirdClefDataset(
        df_train, class_map, mel_shape=mel_shape, augment=True
    )
    val_ds   = BirdClefDataset(
        df_val, class_map, mel_shape=mel_shape, augment=False
    )

    train_loader = create_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ----------------------------------
    # Device setup
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ----------------------------------
    # Checkpoint directory
    # ----------------------------------
    ckpt_root = Path(paths_cfg.get("models_dir", project_root / "models"))
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # Training runs
    # ----------------------------------
    saved_ckpts: list[str] = []

    for run in range(1, num_models + 1):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Linear(in_feats, len(class_map))
        model.to(device)

        # Context for train_model
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
        print(f" • {p}")


if __name__ == "__main__":
    main()

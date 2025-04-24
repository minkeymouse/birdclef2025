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

import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

from src.train.dataloader import BirdClefDataset, create_dataloader, train_model


def main(cfg_path: str) -> None:
    # Load configuration
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Identify EfficientNet block
    arch_cfg = next(
        (a for a in cfg["model"]["architectures"] if a["name"].startswith("efficientnet")),
        None,
    )
    if arch_cfg is None:
        raise ValueError("EfficientNet config block missing in YAML file.")
    model_name = arch_cfg["name"]  # e.g. "efficientnet_b0"
    num_models = arch_cfg.get("num_models", 1)
    pretrained = arch_cfg.get("pretrained", True)

    # ----------------------------------
    # Load and prepare metadata
    # ----------------------------------
    df_meta = pd.read_csv(cfg["dataset"]["train_metadata"])
    if cfg["dataset"].get("include_pseudo", False):
        pseudo_path = Path(cfg["dataset"]["train_metadata"]).with_name("soundscape_metadata.csv")
        if pseudo_path.exists():
            df_meta = pd.concat([df_meta, pd.read_csv(pseudo_path)], ignore_index=True)

    # Build class map from metadata
    if "label_json" in df_meta.columns:
        species = sorted({sp for js in df_meta["label_json"] for sp in json.loads(js).keys()})
    else:
        species = sorted(df_meta["primary_label"].astype(str).unique())
        if "secondary_labels" in df_meta.columns:
            sec = []
            for raw in df_meta["secondary_labels"]:
                if isinstance(raw, str):
                    sec.extend(raw.split())
            species = sorted(set(species + sec))
    class_map = {sp: idx for idx, sp in enumerate(species)}

    # Train/validation split
    val_frac = float(cfg["training"].get("val_fraction", 0.1))
    seed = int(cfg["training"].get("seed", 42))
    df_val = df_meta.sample(frac=val_frac, random_state=seed)
    df_train = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # ----------------------------------
    # Datasets & DataLoaders
    # ----------------------------------
    mel_shape = tuple(cfg["dataset"].get("mel_shape", [128, 256]))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 4))

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
    ckpt_root = Path(cfg.get("paths", {}).get("models_dir", "models")) / model_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("paths", {})["models_dir"] = str(ckpt_root)

    # ----------------------------------
    # Training runs
    # ----------------------------------
    saved_ckpts: list[str] = []
    base_seed = seed

    for run in range(1, num_models + 1):
        torch.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)

        # Instantiate model
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Linear(in_feats, len(class_map))
        model.to(device)

        # Optional warm-start
        init_ckpt = arch_cfg.get("init_checkpoint")
        if init_ckpt:
            ckpt_path = f"{init_ckpt}_{run}.pth"
            if Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
                print(f"Loaded init checkpoint for run {run} from {ckpt_path}")

        # Set context for train_model
        cfg["current_arch"] = model_name
        cfg["current_run"] = run
        cfg["class_map"] = class_map

        # Train and collect checkpoints
        ckpts = train_model(model, train_loader, val_loader, cfg, device)
        saved_ckpts.extend(ckpts)

    # ----------------------------------
    # Summary
    # ----------------------------------
    print("\nTraining complete – saved checkpoints:")
    for p in saved_ckpts:
        print(" •", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 ensemble")
    parser.add_argument(
        "--cfg", type=str, default="config/initial_train.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    main(args.cfg)
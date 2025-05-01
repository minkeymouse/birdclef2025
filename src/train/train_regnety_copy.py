#!/usr/bin/env python3
"""
train_regnety.py — train RegNetY ensemble from scratch using timm-backed RegNetY
====================================================================================

* This version initializes models with ImageNet pretrained weights (pretrained=True)
  and skips loading any previous checkpoints, allowing you to train from scratch.
* Training/paths are controlled via the same YAML used by the EfficientNet trainer (`config/train.yaml`).

Usage
-----
```bash
python -m src.train.train_regnety -c config/train.yaml
```
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
import timm

# ----------------------------------------------------------------------------
# Project imports (repo root two levels up)
# ----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))  # noqa: E402
from src.train.dataloader import BirdClefDataset, create_dataloader, train_model  # noqa: E402
from src.utils import utils  # noqa: E402

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RegNetY ensemble from scratch"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=project_root / "config" / "train.yaml",
        help="Path to YAML training configuration"
    )
    return parser.parse_args()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- config subsections -------------------------------------------------
    dataset_cfg  = cfg["dataset"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    paths_cfg    = cfg["paths"]

    # find the RegNetY entry
    reg_cfg = next(
        a for a in model_cfg.get("architectures", [])
        if a.get("name", "").lower().startswith("regnety")
    )
    model_name = reg_cfg["name"]         # e.g. "regnety_800mf"
    num_runs   = int(reg_cfg.get("num_models", 1))

    # --- load metadata ------------------------------------------------------
    df_meta = pd.read_csv(dataset_cfg["train_metadata"])
    # optionally include pseudo / synthetic
    if dataset_cfg.get("include_pseudo", False):
        pseudo = Path(dataset_cfg["train_metadata"]).with_name("soundscape_metadata.csv")
        if pseudo.exists():
            df_meta = pd.concat([df_meta, pd.read_csv(pseudo)], ignore_index=True)
    if dataset_cfg.get("include_synthetic", False):
        synth = Path(dataset_cfg["train_metadata"]).with_name("synthetic_metadata.csv")
        if synth.exists():
            df_meta = pd.concat([df_meta, pd.read_csv(synth)], ignore_index=True)

    # --- taxonomy -----------------------------------------------------------
    class_list, class_map = utils.load_taxonomy(
        paths_cfg.get("taxonomy_csv"), dataset_cfg.get("train_csv")
    )

    # --- train/val split ----------------------------------------------------
    val_frac  = float(training_cfg.get("val_fraction", 0.1))
    seed_base = int(training_cfg.get("seed", 42))
    df_val    = df_meta.sample(frac=val_frac, random_state=seed_base)
    df_train  = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val    = df_val.reset_index(drop=True)

    # --- datasets and loaders -----------------------------------------------
    mel_shape   = tuple(dataset_cfg.get("target_shape", [256, 256]))
    batch_size  = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))

    train_ds = BirdClefDataset(df_train, class_map, mel_shape=mel_shape, augment=True)
    val_ds   = BirdClefDataset(df_val,   class_map, mel_shape=mel_shape, augment=False)

    train_loader = create_dataloader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader   = create_dataloader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # --- hardware setup -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    saved_ckpts: List[str] = []
    for run_id in range(1, num_runs + 1):
        # reproducibility per run
        torch.manual_seed(seed_base + run_id)
        np.random.seed(seed_base + run_id)

        # create pretrained RegNetY
        model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=3,  # BirdCLEF mel-tensors are 3-channel
            drop_rate=training_cfg.get("drop_rate", 0.0),
            drop_path_rate=training_cfg.get("drop_path_rate", 0.0),
            num_classes=len(class_map),
        )
        # Adapt final layer if timm version separate head
        # (timm.create_model with num_classes updates head automatically)

        model.to(device)

        # update config context for trainer
        cfg["current_arch"] = model_name
        cfg["current_run"]  = run_id
        cfg["class_map"]    = class_map

        # train and collect checkpoints
        ckpts = train_model(model, train_loader, val_loader, cfg, device)
        saved_ckpts.extend(ckpts)

    # --- wrap up ------------------------------------------------------------
    print("\nTraining complete — saved checkpoints:")
    for ck in saved_ckpts:
        print(f" • {ck}")

if __name__ == "__main__":
    main()

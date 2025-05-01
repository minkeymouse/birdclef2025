#!/usr/bin/env python3
"""
train_regnety.py — (re‑)train RegNetY ensemble **from previous checkpoints only**, using timm-backed RegNetY
====================================================================================

* Each run (1‥N) **must** have an existing checkpoint whose filename starts with
  `{model_name}_run{run_id}_`; otherwise the script aborts with FileNotFoundError.
* We resume model weights from that checkpoint but initialise a fresh optimizer/
  scheduler, enabling fine-tuning on updated data while retaining learned features.
* Configuration (dataset, model, training, paths) is read from `config/train.yaml`.

Usage
-----
```bash
python -m src.train.train_regnety -c config/train.yaml [--loss-type {cross_entropy,focal_loss_bce}]
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
# Project imports
# ----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.train.dataloader import (
    BirdClefDataset,
    create_dataloader,
    train_model,
    LOSS_TYPE,
)
from src.utils import utils

# ----------------------------------------------------------------------------
# Helper — locate newest checkpoint for a given run
# ----------------------------------------------------------------------------
def latest_ckpt(run_id: int, ckpt_root: Path, model_name: str) -> Path:
    """Return newest `*.pth` checkpoint or raise FileNotFoundError."""
    pattern = f"{model_name}_run{run_id}_*.pth"
    ckpts = sorted(ckpt_root.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint matching '{pattern}' in {ckpt_root}")
    return ckpts[-1]

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume RegNetY ensemble training from existing checkpoints")
    p.add_argument(
        "--config", "-c",
        type=Path,
        default=project_root / "config" / "train.yaml",
        help="Path to YAML training configuration"
    )
    p.add_argument(
        "--loss-type",
        choices=["cross_entropy", "focal_loss_bce"],
        help="Override the LOSS_TYPE from dataloader.py"
    )
    return p.parse_args()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    # Override loss function if requested
    if args.loss_type:
        CFG.setdefault("training", {})["loss_type"] = args.loss_type

    # Report chosen loss
    final_loss = args.loss_type or LOSS_TYPE
    print(f"→ Using loss function: {final_loss}")

    # Configuration
    dataset_cfg  = CFG["dataset"]
    model_cfg    = CFG["model"]
    training_cfg = CFG["training"]
    paths_cfg    = CFG["paths"]

    # Identify RegNetY architecture
    reg_cfg = next(
        a for a in model_cfg.get("architectures", [])
        if a.get("name", "").lower().startswith("regnety")
    )
    model_name = reg_cfg["name"]
    num_runs   = int(reg_cfg.get("num_models", 1))

    # Load metadata
    df_meta = pd.read_csv(dataset_cfg["train_metadata"], dtype=str, low_memory=False)
    if dataset_cfg.get("include_pseudo", False):
        pseudo = Path(dataset_cfg["train_metadata"]).with_name("soundscape_metadata.csv")
        if pseudo.is_file():
            df_meta = pd.concat([df_meta, pd.read_csv(pseudo, dtype=str)], ignore_index=True)
    if dataset_cfg.get("include_synthetic", False):
        synth = Path(dataset_cfg["train_metadata"]).with_name("synthetic_metadata.csv")
        if synth.is_file():
            df_meta = pd.concat([df_meta, pd.read_csv(synth, dtype=str)], ignore_index=True)

    # Load taxonomy
    _, class_map = utils.load_taxonomy(
        paths_cfg.get("taxonomy_csv"), dataset_cfg.get("train_csv")
    )

    # Split into train/val
    val_frac  = float(training_cfg.get("val_fraction", 0.1))
    seed_base = int(training_cfg.get("seed", 42))
    df_val    = df_meta.sample(frac=val_frac, random_state=seed_base)
    df_train  = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val    = df_val.reset_index(drop=True)

    # Datasets & loaders
    mel_shape   = tuple(dataset_cfg.get("target_shape", [256, 256]))
    batch_size  = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))

    train_ds = BirdClefDataset(df_train, class_map, mel_shape=mel_shape, augment=True)
    val_ds   = BirdClefDataset(df_val,   class_map, mel_shape=mel_shape, augment=False)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = create_dataloader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Checkpoint directory
    ckpt_root = project_root / paths_cfg.get("models_dir", "models")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Training loops for each run
    saved_ckpts: List[str] = []
    for run_id in range(1, num_runs + 1):
        torch.manual_seed(seed_base + run_id)
        np.random.seed(seed_base + run_id)

        # Build RegNetY with 3-channel input and default (untrained) weights
        model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=3,
            num_classes=len(class_map),            # ← ensure head has correct output dim
            drop_rate=float(training_cfg.get("drop_rate", 0.0)),
            drop_path_rate=float(training_cfg.get("drop_path_rate", 0.0)),
        )
        # Adapt final head to correct number of classes
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, len(class_map))
        elif hasattr(model, "classifier"):
            model.classifier = nn.Linear(model.classifier.in_features, len(class_map))

        # Load the most recent checkpoint for this run
        ckpt_path = latest_ckpt(run_id, ckpt_root, model_name)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print(f"✔ Resumed run {run_id} from {ckpt_path.name}")

        model.to(device)

        # Context for trainer
        CFG["current_arch"] = model_name
        CFG["current_run"]  = run_id
        CFG["class_map"]    = class_map

        # Train and collect checkpoints
        ckpts = train_model(model, train_loader, val_loader, CFG, device)
        saved_ckpts.extend(ckpts)

    # Summary
    print("\nTraining complete — saved checkpoints:")
    for ck in saved_ckpts:
        print(f" • {ck}")


if __name__ == "__main__":
    main()

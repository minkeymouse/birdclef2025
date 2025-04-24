#!/usr/bin/env python3
"""
train_regnety.py ─ Ensemble trainer for RegNetY‑800MF
====================================================
* Trains **N** RegNetY‑800MF models (different random seeds) on the BirdCLEF
  10‑second‑chunk dataset using a YAML configuration file.
* Training details (loss, scheduler, checkpointing) are delegated to the shared
  utilities in ``src.train.dataloader`` so this script focuses on orchestration.

Usage
-----
```bash
python src/train/train_regnety.py --cfg config/initial_train.yaml
```
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Local utilities -------------------------------------------------------------
from src.train.dataloader import (
    BirdClefDataset,
    create_dataloader,
    train_model,
)

# -----------------------------------------------------------------------------
# CLI & config loading
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a RegNetY‑800MF ensemble")
    p.add_argument(
        "--cfg",
        default="config/initial_train.yaml",
        help="Path to the YAML configuration file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1) read configuration
    # ------------------------------------------------------------------
    cfg_path = Path(args.cfg)
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    # find the architecture block that starts with "regnet"
    arch_cfg = next(
        (arch for arch in config["model"]["architectures"] if arch["name"].startswith("regnet")),
        None,
    )
    if arch_cfg is None:
        raise ValueError("No RegNetY entry found in the configuration file.")

    model_name: str = arch_cfg["name"]  # e.g. "regnety_800mf"
    n_models: int = arch_cfg.get("num_models", 1)
    use_pretrained: bool = arch_cfg.get("pretrained", True)

    # ------------------------------------------------------------------
    # 2) load metadata CSV (+ optional pseudo‑labeled soundscapes)
    # ------------------------------------------------------------------
    train_meta_path = Path(config["dataset"]["train_metadata"])
    df = pd.read_csv(train_meta_path)

    if config["dataset"].get("include_pseudo", False):
        pseudo_meta = train_meta_path.with_name("soundscape_metadata.csv")
        if pseudo_meta.exists():
            df_pseudo = pd.read_csv(pseudo_meta)
            df = pd.concat([df, df_pseudo], ignore_index=True)

    # ------------------------------------------------------------------
    # 3) build class map (species → index)
    # ------------------------------------------------------------------
    if "label_json" in df.columns:
        species_set = {sp for js in df["label_json"] for sp in json.loads(js).keys()}
    else:
        species_set = set(df["primary_label"].unique())
        if "secondary_labels" in df.columns:
            for sec in df["secondary_labels"]:
                if isinstance(sec, str):
                    species_set.update(sec.split())

    species = sorted(species_set)
    class_map = {sp: idx for idx, sp in enumerate(species)}
    n_classes = len(class_map)

    # ------------------------------------------------------------------
    # 4) train/val split
    # ------------------------------------------------------------------
    val_frac = float(config["training"].get("val_fraction", 0.1))
    df_val = df.sample(frac=val_frac, random_state=42)
    df_train = df.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5) datasets and loaders
    # ------------------------------------------------------------------
    train_ds = BirdClefDataset(df_train, class_map, mel_shape=(128, 256), augment=True)
    val_ds = BirdClefDataset(df_val, class_map, mel_shape=(128, 256), augment=False)

    train_loader = create_dataloader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 6) device and reproducibility
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = int(config["training"].get("seed", 42))

    # output dir: models/regnety_800mf/<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("models") / model_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 7) training loop for N seeds
    # ------------------------------------------------------------------
    saved_ckpts: list[str] = []

    from torchvision import models  # import here to avoid unnecessary dependency if script is only inspected

    for run in range(1, n_models + 1):
        # ------ reproducible seed per run ------
        torch.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)

        # ------ model init ------
        weights = (
            models.RegNet_Y_800MF_Weights.DEFAULT if use_pretrained else None
        )
        model = models.regnet_y_800mf(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model.to(device)

        # ------ optional warm‑start checkpoint ------
        if arch_cfg.get("init_checkpoint"):
            ckpt_path = Path(f"{arch_cfg['init_checkpoint']}_{run}.pth")
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state["model_state_dict"], strict=False)
                print(f"[INIT] Loaded checkpoint {ckpt_path}")

        # ------ per‑run config clone ------
        run_cfg = config.copy()
        run_cfg.update(
            {
                "current_arch": model_name,
                "current_run": run,
                "class_map": class_map,
                "paths": {"models_dir": str(out_dir)},
            }
        )

        # ------ train ------
        best_ckpts = train_model(model, train_loader, val_loader, run_cfg, device)
        saved_ckpts.extend(best_ckpts)

    # ------------------------------------------------------------------
    # 8) summary
    # ------------------------------------------------------------------
    print("\nTraining finished. Top checkpoints:")
    for p in saved_ckpts:
        print(" •", p)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

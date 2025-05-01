#!/usr/bin/env python3
"""
train_regnety.py — (re‑)train RegNetY-ensemble **from previous checkpoints only**, using timm-backed RegNetY
====================================================================================

* For every run `1‥N` the script **must** find an existing checkpoint whose
  filename starts with `{model_name}_run{run_id}_`. Missing checkpoints raise
  *FileNotFoundError* so you never accidentally start from ImageNet weights.
* We reload the *model weights* but **start a fresh optimiser/scheduler** so the
  network is fine‑tuned on the latest dataset/labels without carrying over any
  stale optimiser state.
* Training/paths are controlled via the same YAML used by the EfficientNet
  trainer (`config/train.yaml`).

Usage
-----
```bash
python -m src.train.train_regnety -c config/train.yaml
```
"""
from __future__ import annotations

import argparse
import sys
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
sys.path.insert(0, str(project_root))
from src.train.dataloader import BirdClefDataset, create_dataloader, train_model  # noqa: E402
from src.utils import utils  # noqa: E402

# ----------------------------------------------------------------------------
# Helper — locate newest checkpoint for a given run
# ----------------------------------------------------------------------------

def latest_ckpt(run_id: int, ckpt_root: Path, model_name: str) -> Path:
    """Return newest `*.pth` checkpoint or raise *FileNotFoundError*."""
    pattern = f"{model_name}_run{run_id}_*.pth"
    ckpts = sorted(ckpt_root.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint matching '{pattern}' in {ckpt_root}")
    return ckpts[-1]

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume RegNetY ensemble training")
    p.add_argument(
        "--config", "-c", type=Path,
        default=project_root / "config" / "train.yaml",
        help="Path to YAML training configuration"
    )
    return p.parse_args()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    # --- config excerpts -----------------------------------------------------
    dataset_cfg  = CFG["dataset"]
    model_cfg    = CFG["model"]
    training_cfg = CFG["training"]
    paths_cfg    = CFG["paths"]

    # --- RegNetY block -------------------------------------------------------
    reg_cfg = next(
        a for a in model_cfg["architectures"]
        if a["name"].lower().startswith("regnety")
    )
    model_name = reg_cfg["name"]      # e.g. "regnety_008"
    num_models = int(reg_cfg.get("num_models", 1))

    # --- metadata ------------------------------------------------------------
    df_meta = pd.read_csv(dataset_cfg["train_metadata"])

    # optional: include pseudo / synthetic if flags are set in YAML ------------
    if dataset_cfg.get("include_pseudo", False):
        pseudo_path = Path(dataset_cfg["train_metadata"]).with_name("soundscape_metadata.csv")
        if pseudo_path.is_file():
            df_meta = pd.concat([df_meta, pd.read_csv(pseudo_path)], ignore_index=True)
    if dataset_cfg.get("include_synthetic", False):
        synth_path = Path(dataset_cfg["train_metadata"]).with_name("synthetic_metadata.csv")
        if synth_path.is_file():
            df_meta = pd.concat([df_meta, pd.read_csv(synth_path)], ignore_index=True)

    # taxonomy ---------------------------------------------------------------
    class_list, class_map = utils.load_taxonomy(
        paths_cfg.get("taxonomy_csv"), dataset_cfg.get("train_csv")
    )

    # split -------------------------------------------------------------------
    val_frac  = float(training_cfg.get("val_fraction", 0.1))
    base_seed = int(training_cfg.get("seed", 42))
    df_val    = df_meta.sample(frac=val_frac, random_state=base_seed)
    df_train  = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val    = df_val.reset_index(drop=True)

    # datasets / loaders ------------------------------------------------------
    mel_shape   = tuple(dataset_cfg.get("target_shape", [256, 256]))
    batch_size  = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))

    train_ds = BirdClefDataset(df_train, class_map, mel_shape=mel_shape, augment=True)
    val_ds   = BirdClefDataset(df_val,   class_map, mel_shape=mel_shape, augment=False)

    train_loader = create_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # hardware ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # checkpoint root ---------------------------------------------------------
    ckpt_root = project_root / paths_cfg.get("models_dir", "models")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # train each run ----------------------------------------------------------
    saved_ckpts: List[str] = []
    for run in range(1, num_models + 1):
        torch.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)

        # build model via timm with single-channel input ---------------------
        model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        # adapt final layer
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, len(class_map))
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Linear(
                model.classifier.in_features, len(class_map)
            )

        # load previous checkpoint -------------------------------------------
        ckpt_path = latest_ckpt(run, ckpt_root, model_name)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print(f"✔ Resumed run {run} from {ckpt_path.name}")

        model.to(device)

        # context for generic trainer ----------------------------------------
        CFG["current_arch"] = model_name
        CFG["current_run"]  = run
        CFG["class_map"]    = class_map

        ckpts = train_model(model, train_loader, val_loader, CFG, device)
        saved_ckpts.extend(ckpts)

    # summary ----------------------------------------------------------------
    print("\nTraining complete – saved checkpoints:")
    for p in saved_ckpts:
        print(f" • {p}")


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python3
"""
train_regnety.py — (re‑)train RegNetY-0.8GF ensemble with soft labels (BCEWithLogits only)
==========================================================================================

* Uses timm for model instantiation (RegNetY ensemble).
* Strictly uses BCEWithLogitsLoss for soft-label training (no CE fallback).
* Supports optional MixUp on soft-label vectors.
* Performs stratified K-fold cross-validation, saving best checkpoint per fold.
"""
import argparse
import sys
import time
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import timm
import yaml
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.train.dataloader import BirdClefDataset, collate_fn, create_dataloader
from src.train.train_utils import train_one_epoch, validate

# ----------------------------------------------------------------------------
# Load configuration
# ----------------------------------------------------------------------------
config_path = project_root / "config" / "train.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    full_cfg = yaml.safe_load(f)
cfg = full_cfg["regnety"]

models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# reproducibility
seed = cfg.get("seed", 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Model definition with soft-label BCE and optional MixUp
# ----------------------------------------------------------------------------
class BirdCLEF_REGNETY(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_chans: int,
        num_classes: int,
        pretrained: bool = True,
        mixup_alpha: float = 0.0,
    ):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.mixup_enabled = mixup_alpha > 0.0
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        if self.training and self.mixup_enabled and targets is not None:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[idx]
            y_a, y_b = targets, targets[idx]
            logits = self.backbone(x)
            loss = lam * F.binary_cross_entropy_with_logits(logits, y_a) \
                   + (1 - lam) * F.binary_cross_entropy_with_logits(logits, y_b)
            return logits, loss
        logits = self.backbone(x)
        return logits

# ----------------------------------------------------------------------------
# Factory functions
# ----------------------------------------------------------------------------
def get_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    opt = cfg.get("optimizer", "AdamW")
    lr  = cfg.get("lr", 1e-3)
    wd  = cfg.get("weight_decay", 0.0)
    if opt == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt}")


def get_scheduler(optimizer: optim.Optimizer, cfg: dict):
    sch = cfg.get("scheduler", None)
    if sch == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("T_max", cfg.get("epochs", 20)),
            eta_min=cfg.get("eta_min", 0.0),
        )
    if sch == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.get("eta_min", 0.0),
            verbose=True,
        )
    if sch == 'StepLR':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.get("epochs", 100) // 3),
            gamma=0.5,
        )
    return None


def get_criterion(cfg: dict):
    return nn.BCEWithLogitsLoss()

# ----------------------------------------------------------------------------
# Training over stratified folds
# ----------------------------------------------------------------------------
def run_training():
    # load CSVs
    train_csv = project_root / "data" / "birdclef" / "train.csv"
    meta_csv  = project_root / "data" / "birdclef" / "DATABASE" / "train_metadata.csv"
    train_df    = pd.read_csv(train_csv)
    metadata_df = pd.read_csv(meta_csv)

    # build label2id mapping
    label_list = sorted(train_df["primary_label"].unique())
    label2id   = {lab: i for i, lab in enumerate(label_list)}
    num_classes = len(label_list)

    # attach labels for stratification
    metadata_df["primary_label"] = metadata_df["filename"].map(
        train_df.set_index("filename")["primary_label"]
    )
    y = metadata_df["primary_label"].map(label2id).values

    # StratifiedKFold
    skf = StratifiedKFold(
        n_splits=cfg.get("n_fold", 5), shuffle=True,
        random_state=cfg.get("seed", seed)
    )
    best_scores = []

    # CLI parser for pretrained flag
    args = parse_args()
    pretrained_flag = args.pretrained or cfg.get("pretrained", True)

    # Loop folds
    for fold, (tr_idx, va_idx) in enumerate(skf.split(metadata_df, y)):
        if fold not in cfg.get("selected_folds", list(range(cfg.get("n_fold",5)))):
            continue
        print(f"\n===== Fold {fold} =====")
        train_meta = metadata_df.iloc[tr_idx].reset_index(drop=True)
        val_meta   = metadata_df.iloc[va_idx].reset_index(drop=True)

        # DataLoader
        train_ds = BirdClefDataset(label2id, train_meta, num_classes, mode='train')
        val_ds   = BirdClefDataset(label2id, val_meta,   num_classes, mode='valid')
        train_loader = create_dataloader(
            train_ds,
            batch_size=cfg.get("batch_size",32),
            num_workers=cfg.get("num_workers",4),
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.get("batch_size",32),
            shuffle=False,
            num_workers=cfg.get("num_workers",4),
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Model
        model = BirdCLEF_REGNETY(
            cfg.get("name","regnety_008"),
            cfg.get("in_channels",1),
            num_classes,
            pretrained=pretrained_flag,
            mixup_alpha=cfg.get("mixup_alpha",0.0)
        ).to(device)
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        criterion = get_criterion(cfg)

        best_auc, best_epoch = 0.0, 0

        start_epoch = 0

        ckpt_path = models_dir / f"{cfg['name']}_fold{fold}_best.pth"
        if ckpt_path.exists() and cfg["pretrained"] == False:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            if state.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(state['scheduler_state_dict'])
            start_epoch = state['epoch']
            print(f"Resuming fold {fold} from epoch {start_epoch}")

        for epoch in range(start_epoch, cfg.get("epochs",20)):
            print(f"Epoch {epoch+1}/{int(cfg['epochs'])}")
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # scheduler step
            if scheduler and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_auc > best_auc:
                best_auc, best_epoch = val_auc, epoch+1
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch+1,
                    'val_auc': val_auc,
                }
                save_path = models_dir / f"{cfg['name']}_fold{fold}_best.pth"
                torch.save(ckpt, save_path)

        best_scores.append(best_auc)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print("\n===== CV Results =====")
    for f, score in zip(cfg.get("selected_folds",[]), best_scores):
        print(f"Fold {f}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")

# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RegNetY ensemble with soft labels (BCE only)"
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Init from timm pretrained weights"
    )
    return parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    print("Starting RegNetY training with soft labels...")
    run_training()
    print(f"Done in {(time.time()-start)/60:.2f} min.")

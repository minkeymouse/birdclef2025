#!/usr/bin/env python3
"""
train_efficientnet.py — (re-)train EfficientNet-B0 ensemble from pretrained or previous checkpoints
===========================================================================================

* Each run (1‥N) **must** have an existing checkpoint whose filename starts with
  `{model_name}_run{run_id}_`; unless `--pretrained` is passed, in which case we
  initialize from the timm pretrained model zoo.
* `--pretrained` will skip checkpoint resumption and use pretrained weights.
* Configuration (dataset, model, training, paths) is read from `config/train.yaml`.

Usage
-----
```bash
python -m src.train.train_efficientnet -c config/train.yaml [--pretrained]
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import tqdm
import random

import numpy as np
import pandas as pd
import torch
import yaml

import timm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
config_path  = project_root / "config" / "train.yaml"
sys.path.insert(0, str(project_root))
from src.train.dataloader import BirdClefDataset, collate_fn, create_dataloader
from src.train.train_utils import train_one_epoch, validate, calculate_auc

with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

efficientnet_cfg = CFG["efficientnet"]
efficientnet_cfg = Path(efficientnet_cfg)

# ─── Load metadata ─────────────────────────────────────────
print("Loading taxonomy data…")
taxonomy_df = pd.read_csv("data/birdclef/taxonomy.csv")
species_class_map = dict(zip(taxonomy_df["primary_label"], taxonomy_df["class_name"]))

print("Loading original metadata...")
train_df = pd.read_csv("data/birdclef/train.csv")

# Build your label2id mapping
label_list   = sorted(train_df["primary_label"].unique().tolist())
label2id     = {lab: i for i, lab in enumerate(label_list)}
num_classes  = len(label_list)
print(f"Found {num_classes} unique species labels")
metadata_df = pd.read_csv("data/birdclef/DATABASE/train_metadata.csv")

def latest_ckpt(run_id: int, ckpt_root: Path, model_name: str) -> Path:
    pattern = f"{model_name}_run{run_id}_*.pth"
    ckpts = sorted(ckpt_root.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint matching '{pattern}' in {ckpt_root}")
    return ckpts[-1]

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EfficientNet-B0 ensemble, optionally from pretrained weights"
    )
    p.add_argument(
        "--pretrained", action="store_true",
        help="Initialize models with timm pretrained weights instead of resuming checkpoints"
    )
    return p.parse_args()

class BirdCLEF_EFFICIENTNET(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = efficientnet_cfg
        
        taxonomy_df = taxonomy_df
        cfg["num_classes"] = num_classes
        
        self.backbone = timm.create_model(
            cfg["name"],
            pretrained=cfg["pretrained"],
            in_chans=cfg["in_channels"],
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
            
        self.feat_dim = backbone_out
        
        self.classifier = nn.Linear(backbone_out, cfg["num_classes"])
        
        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg["mixup_alpha > 0"]
        if self.mixup_enabled:
            self.mixup_alpha = cfg["mixup_alpha"]
            
    def forward(self, x, targets=None):
    
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None
        
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        softmax = self.classifier(features)
        
        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.cross_entropy, 
                                       softmax, targets_a, targets_b, lam)
            return softmax, loss
            
        return softmax
    
    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def get_optimizer(model, cfg):
  
    if cfg["optimizer"] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"]
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg["optimizer"]} not implemented")
        
    return optimizer

def get_scheduler(optimizer, cfg):
   
    if cfg["scheduler"] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["T_max"],
            eta_min=cfg["min_lr"]
        )
    elif cfg["scheduler"] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg["min_lr"],
            verbose=True
        )
    elif cfg["scheduler"] == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["epochs"] // 3,
            gamma=0.5
        )
    elif cfg["scheduler"] == 'OneCycleLR':
        scheduler = None  
    else:
        scheduler = None
        
    return scheduler

def get_criterion(cfg):
 
    if cfg["criterion"] == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg["criterion"]} not implemented")
        
    return criterion

def run_training(metadata_df, cfg):

    taxonomy_df = taxonomy_df
    num_classes = num_classes
    df = metadata_df
        
    skf = StratifiedKFold(n_splits=cfg["n_fold"], shuffle=True, random_state=cfg["seed"])
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(:
        if fold not in cfg["selected_folds"]:
            continue
            
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        train_dataset = BirdClefDataset(metadata, label2id, train_df, num_classes, mode='train')
        val_dataset = BirdClefDataset(metadata, label2id, val_df, num_classes, mode='valid')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg["batch_size"], 
            shuffle=True, 
            num_workers=cfg["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg["batch_size"], 
            shuffle=False, 
            num_workers=cfg["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = BirdCLEF_EFFICIENTNET(cfg).to(cfg["device"])
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        if cfg["scheduler"] == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg["lr"],
                steps_per_epoch=len(train_loader),
                epochs=cfg["epochs"],
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        best_auc = 0
        best_epoch = 0
        
        for epoch in range(cfg["epochs"]):
            print(f"\nEpoch {epoch+1}/{cfg["epochs"]}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg["device"],
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, cfg["device"])

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, f"model_fold{fold}.pth")
        
        best_scores.append(best_auc)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg["selected_folds"][fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)

if __name__ == "__main__":
    import time
    
    print("\nLoading training data...")

    print("\nStarting training...")
    
    run_training(, cfg)
    
    print("\nTraining complete!")

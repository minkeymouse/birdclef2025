#!/usr/bin/env python3
"""
train_efficientnet.py – EfficientNet‑B0 ensemble trainer
======================================================
Train **N** EfficientNet‑B0 models (different random seeds) on the BirdCLEF
10‑second‑chunk dataset described in *train_metadata.csv* and a YAML
configuration file (default: ``config/initial_train.yaml``).

Key points implemented from the workflow spec
--------------------------------------------
* **Soft Cross‑Entropy** is used so we can handle one‑hot **and** soft labels
  (primary = 1.0, secondary ≈ 0.05) while still following the user’s wish for
  CE‑style training, not BCE.
* **Sample weighting** – every chunk row can carry a ``weight`` column that is
  multiplied into the loss.
* Top‑3 checkpoints by *macro‑AUC* are stored under
  ``models/efficientnet_b0/`` with time‑stamped filenames.
* Script can be pointed at any YAML cfg via ``--cfg`` – that lets the *automl*
  pipeline reuse the same code for later iterations.

Run example
-----------
```bash
conda activate birdclef
python -m src.train.train_efficientnet \
       --cfg config/initial_train.yaml
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
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

# Project‑local helpers
from src.train.dataloader import BirdClefDataset, create_dataloader
from src.utils.metrics import macro_auc_score, macro_precision_score  # user‑provided util

# ---------------------------------------------------------------------------
# ── Utility: soft‑label cross‑entropy (handles one‑hot + soft labels) ─────────
# ---------------------------------------------------------------------------
class SoftCrossEntropy(nn.Module):
    """Cross‑entropy for *probability* or *soft* targets.

    *   ``input``  – raw logits, shape *(N, C)*
    *   ``target`` – floats in **[0, 1]** of same shape *(N, C)* OR one‑hot.

    The loss is:  *−Σ target × log_softmax(input)* averaged over samples.
    """

    def __init__(self, reduction: str = "mean") -> None:  # reduction ≈ torch style
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(input, dim=1)
        loss = -(target * log_prob).sum(dim=1)  # sum over classes
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()

# ---------------------------------------------------------------------------
# ── Training loop (very thin wrapper around train_utils.train_model logic) ───
# ---------------------------------------------------------------------------

def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    run_id: int,
):
    """Train one seed of EfficientNet and return best checkpoint path list."""

    epochs = cfg["training"]["epochs"]
    opt_cfg = cfg["optimizer"]

    # Optimiser
    optimizer = optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
    )

    # Scheduler (cosine only for now)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["scheduler"].get("T_max", epochs),
        eta_min=cfg["scheduler"].get("eta_min", 1e-6),
    )

    # Loss: soft CE with sample weights applied outside
    criterion = SoftCrossEntropy(reduction="none")

    best_scores: list[float] = []
    best_ckpts: list[str] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            per_sample_loss = criterion(logits, y) * w  # weight each sample
            loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        scheduler.step()

        # Validation
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                preds_all.append(torch.softmax(model(x), dim=1).cpu().numpy())
                targets_all.append(y.cpu().numpy())
        preds_all = np.vstack(preds_all)
        targets_all = np.vstack(targets_all)
        val_auc = macro_auc_score(targets_all, preds_all)
        val_prec = macro_precision_score(targets_all, preds_all, threshold=0.5)

        print(
            f"[Run {run_id}] Epoch {epoch}/{epochs} :: "
            f"loss={epoch_loss/len(train_loader.dataset):.4f} "
            f"valAUC={val_auc:.4f} valPrec@0.5={val_prec:.4f}"
        )

        # Keep top‑3 by AUC
        if len(best_scores) < 3 or val_auc > min(best_scores):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_dir = Path(cfg["paths"]["models_dir"])  # already created by caller
            ckpt_path = ckpt_dir / f"efficientnet_b0_run{run_id}_{ts}.pth"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            if len(best_scores) < 3:
                best_scores.append(val_auc)
                best_ckpts.append(str(ckpt_path))
            else:
                worst = int(np.argmin(best_scores))
                os.remove(best_ckpts[worst])
                best_scores[worst] = val_auc
                best_ckpts[worst] = str(ckpt_path)
            print(f"  ↳ saved checkpoint → {ckpt_path}")
    # Sort checkpoints best→worst before returning
    order = np.argsort(best_scores)[::-1]
    return [best_ckpts[i] for i in order]

# ---------------------------------------------------------------------------
# ── Main entry point ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Locate architecture block for EfficientNet
    arch_cfg = next(
        (a for a in cfg["model"]["architectures"] if a["name"].startswith("efficientnet")),
        None,
    )
    if arch_cfg is None:
        raise ValueError("EfficientNet config block missing in YAML file")

    num_models = arch_cfg.get("num_models", 1)
    pretrained = arch_cfg.get("pretrained", True)

    # ------------------------------------------------------------------
    # Dataset & loaders
    # ------------------------------------------------------------------
    df_meta = pd.read_csv(cfg["dataset"]["train_metadata"])
    # Add pseudo‑labelled rows if requested
    if cfg["dataset"].get("include_pseudo", False):
        pseudo_path = Path(cfg["dataset"]["train_metadata"]).with_name("soundscape_metadata.csv")
        if pseudo_path.exists():
            df_meta = pd.concat([df_meta, pd.read_csv(pseudo_path)], ignore_index=True)

    # Build class map from metadata (label files are npy vectors of fixed length)
    # Here we just infer num_classes from a label sample
    sample_lbl = np.load(df_meta.loc[0, "label_path"])
    num_classes = sample_lbl.shape[0]

    val_frac = cfg["training"].get("val_fraction", 0.1)
    df_val = df_meta.sample(frac=val_frac, random_state=cfg["training"]["seed"])
    df_train = df_meta.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_ds = BirdClefDataset(df_train, mel_shape=(128, 256), augment=True)
    val_ds = BirdClefDataset(df_val, mel_shape=(128, 256), augment=False)

    train_loader = create_dataloader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Directory for checkpoints
    ckpt_root = Path("models") / "efficientnet_b0"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("paths", {})["models_dir"] = str(ckpt_root)

    saved: list[str] = []
    base_seed = cfg["training"]["seed"]

    for run in range(1, num_models + 1):
        torch.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Linear(in_feats, num_classes)
        model.to(device)

        # Optional warm‑start ckpt
        if (ckpt_base := arch_cfg.get("init_checkpoint")) is not None:
            ckpt_path = f"{ckpt_base}_{run}.pth"
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
                print(f"→ loaded init checkpoint for run {run} from {ckpt_path}")

        saved += train_single_model(model, train_loader, val_loader, cfg, device, run)

    print("\nTraining complete – saved checkpoints:")
    for p in saved:
        print(" •", p)


if __name__ == "__main__":
    import yaml  # local import to avoid polluting global namespace

    parser = argparse.ArgumentParser(description="Train EfficientNet ensemble")
    parser.add_argument("--cfg", type=str, default="config/initial_train.yaml", help="Path to YAML config")
    args = parser.parse_args()

    main(args.cfg)

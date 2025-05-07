#!/usr/bin/env python3
"""
inference.py — ensemble inference + ML-KFHE Kalman smoothing

* Loads process.yaml for metadata path.
* Discovers all model checkpoints under models/.
* Applies ensemble inference and multivariate Bernoulli KF smoothing to soundscape chunks.
"""
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from inference_model import InferenceModel, MultivariateBernoulliKalmanFilter

# ─── Load config ────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
config_path  = project_root / "config" / "process.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
# Convert to Path objects
paths_cfg["meta_data"] = Path(paths_cfg["meta_data"])
paths_cfg["mel_dir"]   = Path(paths_cfg["mel_dir"])
paths_cfg["label_dir"] = Path(paths_cfg["label_dir"])

# Models directory
models_dir = project_root / "models"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

def load_ensemble(models_dir: Path, num_classes: int, device: torch.device) -> list:
    models = []
    for ckpt in sorted(models_dir.glob("*.pth")):
        name = ckpt.stem.lower()
        if "efficientnet" in name:
            arch = 'efficientnet_b0'
        elif "regnety" in name:
            arch = 'regnety_008'
        else:
            logger.warning("Skipping unknown checkpoint: %s", ckpt.name)
            continue

        model = InferenceModel(arch, in_chans=1, num_classes=num_classes).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        models.append(model)
        logger.info("Loaded checkpoint: %s as %s", ckpt.name, arch)
    return models


def update_labels_for_group(df_group: pd.DataFrame, models: list, device: torch.device) -> None:
    df = df_group.sort_values('end_sec').reset_index(drop=True)
    # stack mel-spectrograms
    specs = [np.load(p) for p in df['mel_path']]
    specs = np.stack([s if s.ndim==3 else s[None,...] for s in specs], axis=0)

    # ensemble predictions
    all_preds = []
    batch = torch.tensor(specs, dtype=torch.float32, device=device)
    with torch.no_grad():
        for model in models:
            all_preds.append(model(batch).cpu().numpy())
    all_preds = np.stack(all_preds, axis=0)  # (M, T, C)
    M, T, C = all_preds.shape
    ensemble = all_preds.mean(axis=0)       # (T, C)

    # Kalman Filter for probabilities
    kf = MultivariateBernoulliKalmanFilter(
        num_states=C,
        Q_method="kappa",
        kappa=0.05,
        constant_q=1e-3,
        include_offdiag=False,
        r_min=1e-3,
        missing_tau=0.01
    )
    kf.compute_R(ensemble)
    kf.compute_Q()
    kf.initialize(ensemble[0], P0=1e6)

    # Kalman Filter for model trust
    kf_w = MultivariateBernoulliKalmanFilter(
        num_states=M,
        Q_method="constant",
        constant_q=1e-2,
        include_offdiag=False,
        r_min=1e-3,
        missing_tau=0.0
    )
    init_trust = np.full(M, 0.5, dtype=np.float32)
    kf_w.initialize(init_trust, P0=1e2)

    # smoothing loop
    for t in range(T):
        kf.predict()
        kf_w.predict()
        state_pred = kf.probabilities  # (C,)

        # update trust
        residuals = np.mean(np.abs(all_preds[:,t,:] - state_pred), axis=1).astype(np.float32)
        kf_w.update(residuals)
        trust = kf_w.probabilities  # (M,)

        # sequential measurement updates
        for m in range(M):
            prev = all_preds[m, t-1] if t>0 else all_preds[m, t]
            curr = all_preds[m, t]
            nxt  = all_preds[m, t+1] if t<T-1 else all_preds[m, t]
            z_tm = 0.2*prev + 0.6*curr + 0.2*nxt
            z_he = 0.5*z_tm + 0.5*state_pred
            base_R = kf.Ry.diagonal()
            R_eff  = base_R / (trust[m] + 1e-3)
            kf.update(z_he, obs_var=R_eff)

        # save smoothed
        sm = kf.probabilities
        label_path = Path(df.loc[t, 'label_path'])
        np.save(label_path, sm)
        logger.info("Updated labels for %s", label_path.name)


def main():
    logger.info("Starting inference and smoothing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # load metadata
    meta_df = pd.read_csv(paths_cfg["meta_data"])
    sc_df   = meta_df[meta_df['source']=="train_soundscape"]
    if sc_df.empty:
        logger.warning("No soundscape entries found in metadata.")
        return

    # infer num_classes
    num_classes = int(np.load(sc_df.iloc[0]['label_path']).shape[0])

    # load ensemble
    models = load_ensemble(models_dir, num_classes, device)
    if not models:
        logger.error("No models loaded. Exiting.")
        return

    # group by file
    for fname, grp in sc_df.groupby('filename'):
        logger.info("Processing soundscape: %s (%d chunks)", fname, len(grp))
        update_labels_for_group(grp, models, device)

    logger.info("Inference complete.")

if __name__ == "__main__":
    main()

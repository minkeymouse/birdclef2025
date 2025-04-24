#!/usr/bin/env python3
"""
Inference script for BirdCLEF-2025
=================================
Runs chunk-level prediction on every audio file found under
`paths.test_audio_dir` and writes a submission-ready CSV whose columns
match the species order in `paths.taxonomy_csv`.

This file is **self-contained** – the helper routines that used to live in
`src.utils.inference_utils` are embedded at the bottom so you can execute
this script standalone in any environment that has *torch*, *librosa*,
*opencv-python* and *pandas* installed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BirdCLEF-2025 inference")
    p.add_argument("--cfg", default="config/inference.yaml", help="YAML config file")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch", type=int, default=None, help="override batch size")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: load taxonomy → ordered class list
# ---------------------------------------------------------------------------

def _load_species(tax_csv: Path) -> List[str]:
    if not tax_csv.is_file():
        raise FileNotFoundError(f"taxonomy csv not found: {tax_csv}")
    df = pd.read_csv(tax_csv)
    for col in ("primary_label", "ebird_code", "species_code"):
        if col in df.columns:
            return list(df[col])
    raise ValueError("No recognised label column in taxonomy CSV")


# ---------------------------------------------------------------------------
# Inference entry-point
# ---------------------------------------------------------------------------

def main():
    args = _cli()

    # --------------------- load YAML -----------------------------
    with Path(args.cfg).open() as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg["paths"]
    inf_cfg = cfg["inference"]

    audio_dir = Path(paths_cfg["test_audio_dir"]).expanduser()
    models_dir = Path(paths_cfg["models_dir"]).expanduser()
    out_csv = Path(paths_cfg["output_file"]).expanduser()
    tax_csv = Path(paths_cfg["taxonomy_csv"]).expanduser()

    species = _load_species(tax_csv)
    n_classes = len(species)

    chunk_sec = float(inf_cfg["chunk_duration"])
    hop_sec = float(inf_cfg["chunk_hop"])
    batch_size = args.batch or int(inf_cfg.get("batch_size", 32))
    smooth_k = int(inf_cfg.get("smoothing_neighbors", 2))
    presence_thr = float(inf_cfg.get("presence_threshold", 0.5))  # not used for CSV, kept for debug

    ensemble_ckpts: List[str] = list(cfg["ensemble"]["checkpoints"])
    ens_strategy = cfg["ensemble"].get("strategy", "average")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("infer")
    log.info("Device: %s", device)

    # --------------------- load models ---------------------------
    models: List[torch.nn.Module] = []
    arch_tags: List[str] = []
    for ckpt_name in ensemble_ckpts:
        if ckpt_name.startswith("efficientnet"):
            arch = "efficientnet_b0"
        elif ckpt_name.startswith("regnety"):
            arch = "regnety_800mf"
        else:
            raise ValueError(f"Cannot infer architecture from {ckpt_name}")

        ckpt_path = models_dir / arch / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)

        model = _load_model(arch, n_classes, ckpt_path, device)
        models.append(model)
        arch_tags.append(arch)
        log.info("loaded %s", ckpt_path.name)

    # --------------------- iterate audio ------------------------
    results: List[Dict[str, float]] = []
    for wav_path in sorted(audio_dir.iterdir()):
        if wav_path.suffix.lower() not in {".wav", ".ogg", ".mp3"}:
            continue
        log.info("Soundscape: %s", wav_path.name)
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        hop_samples = int(hop_sec * sr)
        chunk_samples = int(chunk_sec * sr)
        pad = (hop_samples - len(y) % hop_samples) % hop_samples
        if pad:
            y = np.pad(y, (0, pad))
        starts = np.arange(0, len(y) - chunk_samples + 1, hop_samples)
        n_chunks = len(starts)

        # per-model probabilities ------------------------------------------------
        per_model: List[np.ndarray] = []  # shape: (n_chunks, n_classes)
        for model in models:
            probs = np.zeros((n_chunks, n_classes), dtype=np.float32)
            for bi in tqdm(range(0, n_chunks, batch_size), desc=model.__class__.__name__, leave=False):
                idx = starts[bi : bi + batch_size]
                waves = [y[s : s + chunk_samples] for s in idx]
                mels = [_wave_to_mel(w, sr) for w in waves]
                x = (
                    torch.from_numpy(np.stack(mels))
                    .unsqueeze(1)
                    .repeat(1, 3, 1, 1)
                    .to(device)
                )
                with torch.no_grad():
                    logits = model(x)
                    probs_batch = torch.softmax(logits, 1).cpu().numpy()
                probs[bi : bi + len(waves)] = probs_batch
            per_model.append(probs)

        # smoothing -------------------------------------------------------------
        if smooth_k > 0:
            per_model = [_smooth(p, 2 * smooth_k + 1) for p in per_model]

        combined = _ensemble(per_model, strategy=ens_strategy)

        # write rows ------------------------------------------------------------
        file_id = wav_path.stem
        for i in range(n_chunks):
            row: Dict[str, float] = {"row_id": f"{file_id}_{i * int(hop_sec)}"}
            row.update({sp: float(combined[i, j]) for j, sp in enumerate(species)})
            results.append(row)

    # --------------------- save CSV ------------------------------
    df_out = pd.DataFrame(results)
    df_out = df_out[["row_id", *species]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    log.info("saved %s (%d rows)", out_csv, len(df_out))


# ===========================================================================
# Embedded helpers (previously src.utils.inference_utils)                     
# ===========================================================================

def _load_model(arch: str, n_classes: int, ckpt_path: Path, device: torch.device):
    """Create model architecture and load checkpoint (pth OR torchscript)."""
    if ckpt_path.suffix in {".ts", ".pt", ".jit", ".pth"} and ckpt_path.name.endswith(".ts.pt"):
        model = torch.jit.load(str(ckpt_path), map_location=device)
        return model.eval()

    from torchvision import models as tvm

    if arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, n_classes)
    elif arch == "regnety_800mf":
        model = tvm.regnet_y_800mf(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    else:
        raise ValueError(arch)

    state = torch.load(ckpt_path, map_location=device)
    key = "model_state_dict" if "model_state_dict" in state else "state_dict"
    model.load_state_dict(state[key])
    model.to(device)
    model.eval()
    return model


def _wave_to_mel(wave: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        wave,
        sr=sr,
        n_fft=1024,
        hop_length=500,
        n_mels=128,
        fmin=40,
        fmax=15000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return cv2.resize(mel_db, (256, 128))


def _smooth(prob: np.ndarray, win: int = 5) -> np.ndarray:
    if win < 2:
        return prob
    k = win // 2
    out = np.zeros_like(prob)
    for i in range(prob.shape[0]):
        out[i] = prob[max(0, i - k) : i + k + 1].mean(0)
    return out


def _ensemble(preds: List[np.ndarray], strategy: str = "average") -> np.ndarray:
    if strategy == "average":
        return np.mean(np.stack(preds), 0)
    elif strategy == "min_then_avg":
        half = len(preds) // 2
        a1 = np.minimum.reduce(preds[:half])
        a2 = np.minimum.reduce(preds[half:])
        return (a1 + a2) / 2.0
    else:
        raise ValueError(strategy)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

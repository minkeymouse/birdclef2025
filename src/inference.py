#!/usr/bin/env python3
"""
inference.py – BirdCLEF-2025 final inference
===========================================
Loads trained checkpoints to run chunk-level predictions on all test audio
and emits a submission CSV matching the taxonomy order.

Assumes: torch, librosa, cv2, pandas, numpy, yaml, tqdm installed,
and project utilities under src/utils.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np
import pandas as pd
import torch
import librosa
import cv2
from tqdm import tqdm

from src.utils.utils import load_taxonomy, resize_mel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF-2025 inference")
    parser.add_argument(
        "--cfg", type=str, default="config/inference.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size from config"
    )
    return parser.parse_args()


def load_models(
    ckpt_names: List[str],
    models_dir: Path,
    n_classes: int,
    device: torch.device,
) -> List[torch.nn.Module]:
    models = []
    for name in ckpt_names:
        # infer architecture from filename prefix
        if name.startswith("efficientnet"):
            arch = "efficientnet_b0"
        elif name.startswith("regnety"):
            arch = "regnety_800mf"
        else:
            raise ValueError(f"Unknown architecture for checkpoint {name}")

        ckpt_path = models_dir / arch / name
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if ckpt_path.suffix == ".pt" and name.endswith(".ts.pt"):
            model = torch.jit.load(str(ckpt_path), map_location=device)
        else:
            from torchvision import models as tvm
            if arch == "efficientnet_b0":
                model = tvm.efficientnet_b0(weights=None)
                model.classifier[1] = torch.nn.Linear(
                    model.classifier[1].in_features, n_classes
                )
            else:
                model = tvm.regnet_y_800mf(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
            state = torch.load(ckpt_path, map_location=device)
            key = "model_state_dict" if "model_state_dict" in state else "state_dict"
            model.load_state_dict(state[key])
        model = model.to(device).eval()
        models.append(model)
    return models


def wave_to_mel(
    wave: np.ndarray,
    sr: int,
    mel_cfg: Dict[str, float],
) -> np.ndarray:
    m = librosa.feature.melspectrogram(
        wave,
        sr=sr,
        n_fft=int(mel_cfg.get("n_fft", 1024)),
        hop_length=int(mel_cfg.get("hop_length", 512)),
        n_mels=int(mel_cfg.get("n_mels", 128)),
        fmin=mel_cfg.get("fmin", 20),
        fmax=mel_cfg.get("fmax", sr // 2),
        power=mel_cfg.get("power", 2.0),
    )
    mel_db = librosa.power_to_db(m, ref=np.max)
    # resize to (n_mels, time_frames)
    target_h = int(mel_cfg.get("target_h", mel_cfg.get("n_mels", 128)))
    target_w = int(mel_cfg.get("target_w", mel_cfg.get("hop_length", 512) * 2))
    return resize_mel(mel_db, target_h, target_w)


def main() -> None:
    args = parse_args()
    # Load config
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    paths = cfg["paths"]
    inf_cfg = cfg.get("inference", {})
    mel_cfg = cfg.get("mel", {})

    # Setup
    audio_dir = Path(paths["test_audio_dir"]).expanduser()
    models_dir = Path(paths["models_dir"]).expanduser()
    output_csv = Path(paths["output_file"]).expanduser()
    taxonomy_csv = Path(paths["taxonomy_csv"]).expanduser()

    class_list, _ = load_taxonomy(taxonomy_csv, None)
    n_classes = len(class_list)

    chunk_sec = float(inf_cfg.get("chunk_duration", 10.0))
    hop_sec = float(inf_cfg.get("chunk_hop", 5.0))
    batch_size = args.batch or int(inf_cfg.get("batch_size", 32))
    smooth_k = int(inf_cfg.get("smoothing_neighbors", 0))

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("inference")
    log.info("Running on device: %s", device)

    # Load models
    ckpts = cfg["ensemble"]["checkpoints"]
    strategy = cfg["ensemble"].get("strategy", "average")
    models = load_models(ckpts, models_dir, n_classes, device)

    results: List[Dict[str, float]] = []
    # Process each test file
    for audio_path in sorted(audio_dir.glob("*.*")):
        if audio_path.suffix.lower() not in {".wav", ".ogg", ".mp3"}:
            continue
        log.info("Processing %s", audio_path.name)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        hop_samples = int(hop_sec * sr)
        chunk_samples = int(chunk_sec * sr)
        pad = (hop_samples - len(y) % hop_samples) % hop_samples
        if pad:
            y = np.pad(y, (0, pad))
        starts = np.arange(0, len(y) - chunk_samples + 1, hop_samples)

        # Collect model probabilities
        preds_per_model = []
        for model in models:
            probs = np.zeros((len(starts), n_classes), dtype=np.float32)
            for i in tqdm(range(0, len(starts), batch_size), desc=model.__class__.__name__, leave=False):
                idx = starts[i : i + batch_size]
                waves = [y[s : s + chunk_samples] for s in idx]
                mels = np.stack([wave_to_mel(w, sr, mel_cfg) for w in waves])
                x = torch.from_numpy(mels).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
                probs[i : i + len(idx)] = probs_batch
            preds_per_model.append(probs)

        # Optional smoothing
        if smooth_k > 0:
            window = 2 * smooth_k + 1
            smoothed = []
            for p in preds_per_model:
                out = np.copy(p)
                for t in range(p.shape[0]):
                    start = max(0, t - smooth_k)
                    end = min(p.shape[0], t + smooth_k + 1)
                    out[t] = p[start:end].mean(axis=0)
                smoothed.append(out)
            preds_per_model = smoothed

        # Ensemble fusion
        stack = np.stack(preds_per_model)
        if strategy == "average":
            combined = np.mean(stack, axis=0)
        elif strategy == "min_then_avg":
            half = len(preds_per_model) // 2
            a1 = np.minimum.reduce(preds_per_model[:half])
            a2 = np.minimum.reduce(preds_per_model[half:])
            combined = (a1 + a2) / 2.0
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")

        # Write rows
        file_id = audio_path.stem
        for t, probs in enumerate(combined):
            row = {"row_id": f"{file_id}_{int(t * hop_sec)}"}
            row.update({species: float(probs[i]) for i, species in enumerate(class_list)})
            results.append(row)

    # Save submission
    df_out = pd.DataFrame(results)
    df_out = df_out[["row_id", *class_list]]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    log.info("Saved %s with %d rows", output_csv, len(df_out))


if __name__ == "__main__":
    main()

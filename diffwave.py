#!/usr/bin/env python3
"""
diffwave.py – inference-only minority-class synthesis
====================================================
Uses a pretrained DiffWave vocoder to convert precomputed
mel-spectrogram chunks (from process.py) into 5-second waveforms.
Usage:
  python diffwave.py generate [--species S1,S2]
  python diffwave.py remove
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from speechbrain.inference.vocoders import DiffWaveVocoder
from configure import CFG

# ────────────────────────────────────────────────────────────
# Config & Hyperparameters
# ────────────────────────────────────────────────────────────
SAMPLE_RATE = 32_000
SEG_SECONDS = 5                                  # Clip length (s)
N_MELS      = 80                                 # Must match vocoder config
HOP_LENGTH  = 256                                # Must match vocoder config

THRESH_LOW, THRESH_HIGH = 20, 50
TARGET_LOW, TARGET_MID   = 20, 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def compute_plan(csv_path: Path) -> Dict[str,int]:
    """Decide how many synthetic clips per species."""
    df = pd.read_csv(csv_path)
    counts = df["primary_label"].value_counts()
    plan: Dict[str,int] = {}
    for sp, cnt in counts.items():
        if cnt < THRESH_LOW:
            plan[sp] = TARGET_LOW - cnt
        elif cnt < THRESH_HIGH:
            plan[sp] = TARGET_MID
        else:
            plan[sp] = 0
    return plan


def patch_train_csv(rows: List[Tuple[str,str]]) -> None:
    """Append new synthetic entries to train.csv."""
    df = pd.read_csv(CFG.TRAIN_CSV)
    exist = set(zip(df["primary_label"], df["filename"]))
    new = [r for r in rows if r not in exist]
    if not new:
        return
    extra = pd.DataFrame(new, columns=["primary_label","filename"])
    for c in df.columns:
        if c not in extra.columns:
            extra[c] = pd.NA
    out = pd.concat([df, extra[df.columns]], ignore_index=True)
    out.to_csv(CFG.TRAIN_CSV, index=False)


def remove_synthetic() -> None:
    """Delete all synthetic .ogg files."""
    for spdir in CFG.TRAIN_AUDIO_DIR.iterdir():
        if spdir.is_dir():
            for f in spdir.glob("synthetic_*.ogg"):
                f.unlink(missing_ok=True)

# ────────────────────────────────────────────────────────────
# Generation: Mel→Wave conversion
# ────────────────────────────────────────────────────────────
def generate(args: argparse.Namespace) -> None:
    """Generate waveforms from precomputed mel-spectrogram files."""
    vb = DiffWaveVocoder.from_hparams(
        source="speechbrain/tts-diffwave-ljspeech",
        savedir=CFG.DIFFWAVE_MODEL_DIR/"vocoder",
        run_opts={"device": DEVICE.type},
    )

    plan = compute_plan(CFG.TRAIN_CSV)
    rows: List[Tuple[str,str]] = []
    base = CFG.PROCESSED_DIR / "mels" / "train"

    for sp, n in plan.items():
        if n <= 0 or (args.species and sp not in args.species):
            continue
        sp_dir = base / sp
        if not sp_dir.exists():
            continue
        mel_files = sorted(sp_dir.glob("*.npy"))
        count = 0

        for mel_fp in mel_files:
            if count >= n:
                break
            mel_np = np.load(mel_fp)
            mel_tensor = torch.from_numpy(mel_np).unsqueeze(0).to(DEVICE)

            wav_tensor = vb.decode_spectrogram(
                mel_tensor,
                hop_length=HOP_LENGTH,
                fast_sampling=True,
                fast_sampling_noise_schedule=[0.0001,0.001,0.01,0.05,0.2,0.5],
            )  # returns [1, time]

            out_fn = f"synthetic_{count:03d}.ogg"
            out_fp = CFG.TRAIN_AUDIO_DIR / sp / out_fn
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(out_fp),
                wav_tensor.cpu().numpy(),
                SAMPLE_RATE,
                format="OGG",
                subtype="VORBIS",
            )

            rows.append((sp, out_fn))
            count += 1

    if rows:
        patch_train_csv(rows)

# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diffwave.py",
        description="DiffWave inference-only synthesizer",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="Generate synthetic audio from mel files")
    gen.add_argument(
        "--species", type=lambda s: s.split(","),
        help="Comma-separated list of species to target",
    )
    sub.add_parser("remove", help="Remove all synthetic audio files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "generate":
        generate(args)
    else:
        remove_synthetic()


if __name__ == "__main__":
    main()

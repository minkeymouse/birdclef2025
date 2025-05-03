#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import logging
import sys
from pathlib import Path
from typing import List
import ast

import math
import cv2
import time
import os
import librosa
import numpy as np
import pandas as pd
import yaml
import tqdm

project_root = Path(__file__).resolve().parents[2]
config_path  = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils import utils

# ─── Load configs ──────────────────────────────────────────
with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
audio_cfg = CFG["audio"]
debug_cfg = CFG["debug"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_gold")

print(f"Debug mode: {'ON' if debug_cfg["enabled"] else 'OFF'}")
print(f"Max samples to process: {debug_cfg["N_MAX"] if debug_cfg["N_MAX"] is not None else 'ALL'}")

# ─── Load metadata ─────────────────────────────────────────
print("Loading taxonomy data…")
taxonomy_df = pd.read_csv(f'{paths_cfg["DATA_ROOT"]}/taxonomy.csv')
# Map species string → class name (if needed)
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

print("Loading training metadata…")
train_df = pd.read_csv(f'{paths_cfg["DATA_ROOT"]}/train.csv')

# Build your label2id mapping from unique primary_label values:
label_list = sorted(train_df['primary_label'].unique().tolist())
label2id   = {label: idx for idx, label in enumerate(label_list)}
num_classes = len(label_list)
print(f"Found {num_classes} unique species labels")

# ─── Subset & prep DataFrame ───────────────────────────────
working_df = train_df[['primary_label', 'secondary_labels', 'rating', 'filename']].copy()
working_df['filepath'] = paths_cfg["DATA_ROOT"] + '/train_audio/' + working_df['filename']

total_samples = min(len(working_df), debug_cfg["N_MAX"] or len(working_df))
print(f"Will process up to {total_samples} samples")

# ─── Utils ─────────────────────────────────────────────────
def audio2melspec(audio_data):
    """Convert a 1D waveform to a normalized mel-spectrogram."""
    if np.isnan(audio_data).any():
        audio_data = np.nan_to_num(audio_data, nan=np.nanmean(audio_data))

    m = librosa.feature.melspectrogram(
        y=audio_data,
        sr=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        fmin=audio_cfg["fmin"],
        fmax=audio_cfg["fmax"],
        power=2.0
    )
    m_db = librosa.power_to_db(m, ref=np.max)
    return (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)

def parse_secondary(s):
    if pd.isna(s) or s in ['', "[]", "['']"]:
        return []
    return ast.literal_eval(s)

# ─── Main processing ────────────────────────────────────────
print("Starting audio processing…")
meta_rows: List[dict] = []
errors = []

for i, row in tqdm.tqdm(working_df.iterrows(), total=total_samples):
    if debug_cfg["enabled"] and i >= debug_cfg["N_MAX"]:
        break

    fname = row.filename
    path  = row.filepath
    sec_labels = parse_secondary(row.secondary_labels)
    file_weight = (row.rating - 5) / (5 - row.rating)
    # Build your weighted one-hot vector
    onehot = np.zeros(num_classes, dtype=np.float32)
    pid    = label2id[row.primary_label]      # primary is already the raw value
    onehot[pid] = 1.0
    for sl in sec_labels:
        if sl in label2id:
            onehot[label2id[sl]] = 0.3

    # Load & chunk the audio
    if not os.path.exists(path):
        log.warning("Missing file: %s", path)
        continue

    try:
        wav, _ = librosa.load(path, sr=audio_cfg["sample_rate"])
        wav, _ = librosa.effects.trim(wav, top_db=audio_cfg["trim_top_db"])
        if wav.size == 0:
            continue

        # If too short, tile/pad to exactly `train_duration`
        dur_samples = int(audio_cfg["train_duration"] * audio_cfg["sample_rate"])
        if wav.size < dur_samples:
            reps = int(math.ceil(dur_samples / wav.size))
            wav = np.tile(wav, reps)[:dur_samples]

        ptr = 0
        hop = int(audio_cfg["train_chunk_hop"] * audio_cfg["sample_rate"])
        while ptr + dur_samples <= wav.size:
            chunk = wav[ptr : ptr + dur_samples]
            ptr += hop

            # Skip silent or speech‐detected chunks if needed
            if utils.is_silent(chunk, db_thresh=audio_cfg["silence_thresh_db"]):
                continue
            if utils.contains_voice(chunk, audio_cfg["sample_rate"]):
                continue

            # Convert to mel-spec, resize, save
            m = audio2melspec(chunk)
            if m.shape != tuple(audio_cfg["target_shape"]):
                m = cv2.resize(m, tuple(audio_cfg["target_shape"]), interpolation=cv2.INTER_LINEAR)
            m = m.astype(np.float32)

            chunk_id    = utils.hash_chunk_id(fname, ptr / audio_cfg["sample_rate"])
            m_path      = paths_cfg["mel_dir"]   / f"{chunk_id}.npy"
            label_path  = paths_cfg["label_dir"] / f"{chunk_id}.npy"

            np.save(m_path, m)
            np.save(label_path, onehot)

            meta_rows.append(
                {
                    "filename": fname,
                    "end_sec": round(ptr / audio_cfg["sample_rate"], 3),
                    "mel_path": str(m_path),
                    "label_path": str(label_path),
                    "weight": float(file_weight),
                }
            )

    except Exception as e:
        errors.append((path, str(e)))

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(paths_cfg["meta_data"], index=False)
end_time = time.time()
print(f"Encountered {len(errors)} errors")
print(errors)
print(f"Saved {len(meta_rows)} chunks to metadata, {len(errors)} errors")

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

# Convert path-strings to Path objects so “/” works
paths_cfg["DATA_ROOT"]  = Path(paths_cfg["DATA_ROOT"])
paths_cfg["audio_dir"]  = Path(paths_cfg["audio_dir"])
paths_cfg["mel_dir"]    = Path(paths_cfg["mel_dir"])
paths_cfg["label_dir"]  = Path(paths_cfg["label_dir"])
paths_cfg["meta_data"]  = Path(paths_cfg["meta_data"])

# Make sure output dirs exist
paths_cfg["mel_dir"].mkdir(parents=True, exist_ok=True)
paths_cfg["label_dir"].mkdir(parents=True, exist_ok=True)
paths_cfg["meta_data"].parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_gold")

print(f"Debug mode: {'ON' if debug_cfg['enabled'] else 'OFF'}")

# ─── Load metadata ─────────────────────────────────────────
print("Loading taxonomy data…")
taxonomy_df = pd.read_csv(paths_cfg["DATA_ROOT"] / "taxonomy.csv")
species_class_map = dict(zip(taxonomy_df["primary_label"], taxonomy_df["class_name"]))

print("Loading training metadata…")
train_df = pd.read_csv(paths_cfg["DATA_ROOT"] / "train.csv")

# Build your label2id mapping
label_list   = sorted(train_df["primary_label"].unique().tolist())
label2id     = {lab: i for i, lab in enumerate(label_list)}
num_classes  = len(label_list)
print(f"Found {num_classes} unique species labels")

# ─── prep DataFrame ────────────────────────────────────────
working_df = train_df[["primary_label", "secondary_labels", "rating", "filename"]].copy()
working_df["filepath"] = paths_cfg["audio_dir"] / working_df["filename"]

total_samples = min(len(working_df), debug_cfg["N_MAX"] or len(working_df))
print(f"Will process up to {total_samples} samples")

# ─── Parse/normalize config values ─────────────────────────
# target_shape is now a YAML list [h, w]
TARGET_H, TARGET_W = audio_cfg["target_shape"]
db_thresh         = audio_cfg.get("silence_thresh_db", -50.0)
dur_samples       = int(audio_cfg["train_duration"] * audio_cfg["sample_rate"])
hop_samples       = int(audio_cfg["train_chunk_hop"] * audio_cfg["sample_rate"])

# ─── Helper functions ──────────────────────────────────────
def audio2melspec(wav: np.ndarray) -> np.ndarray:
    if np.isnan(wav).any():
        wav = np.nan_to_num(wav, nan=np.nanmean(wav))
    m = librosa.feature.melspectrogram(
        y=wav,
        sr=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        fmin=audio_cfg["fmin"],
        fmax=audio_cfg["fmax"],
        power=2.0,
    )
    m_db = librosa.power_to_db(m, ref=np.max)
    return (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)

def parse_secondary(s) -> List[str]:
    if pd.isna(s) or s in ["", "[]", "['']"]:
        return []
    return ast.literal_eval(s)

# ─── Main loop ─────────────────────────────────────────────
meta_rows: List[dict] = []
errors:    List[tuple] = []

# 1) Decide how many rows to actually process
if debug_cfg["enabled"]:
    df_proc = working_df.head(debug_cfg["N_MAX"])
else:
    df_proc = working_df

iterator = df_proc.iterrows()
if not debug_cfg["enabled"]:
    iterator = tqdm.tqdm(iterator, total=len(df_proc),
                         desc="Audio chunks")

for i, row in iterator:
    # if we're in debug mode, stop at N_MAX
    if debug_cfg["enabled"] and i >= debug_cfg["N_MAX"]:
        break

    path       = row.filepath
    fname      = row.filename
    sec_labels = parse_secondary(row.secondary_labels)
    file_weight = file_weight = float(row.rating) / 5.0

    # build weighted one-hot
    onehot = np.zeros(num_classes, dtype=np.float32)
    pid    = label2id[row.primary_label]
    onehot[pid] = 1.0
    for sl in sec_labels:
        if sl in label2id:
            onehot[label2id[sl]] = 0.3

    if not path.exists():
        log.warning("Missing file: %s", path)
        continue

    try:
        wav, _ = librosa.load(path, sr=audio_cfg["sample_rate"])
        wav, _ = librosa.effects.trim(wav, top_db=audio_cfg["trim_top_db"])
        if wav.size == 0:
            continue

        # pad/ tile up to one chunk if short
        if wav.size < dur_samples:
            reps = math.ceil(dur_samples / wav.size)
            wav  = np.tile(wav, reps)[:dur_samples]

        ptr = 0
        while ptr + dur_samples <= wav.size:
            chunk = wav[ptr : ptr + dur_samples]
            ptr  += hop_samples

            if utils.is_silent(chunk, db_thresh=db_thresh):
                continue
            if utils.contains_voice(chunk, audio_cfg["sample_rate"]):
                continue

            m = audio2melspec(chunk)
            if m.shape != (TARGET_H, TARGET_W):
                # cv2.resize takes (width, height)
                m = cv2.resize(m, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            m = m.astype(np.float32)

            chunk_id   = utils.hash_chunk_id(fname, ptr / audio_cfg["sample_rate"])
            m_path     = paths_cfg["mel_dir"]   / f"{chunk_id}.npy"
            label_path = paths_cfg["label_dir"] / f"{chunk_id}.npy"

            np.save(m_path, m)
            np.save(label_path, onehot)

            meta_rows.append({
                "filename":   fname,
                "end_sec":    round(ptr / audio_cfg["sample_rate"], 3),
                "mel_path":   str(m_path),
                "label_path": str(label_path),
                "weight":     file_weight,
                "source": str("train_audio"),
            })

    except Exception as e:
        errors.append((str(path), str(e)))

# ─── Write out metadata ────────────────────────────────────
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(paths_cfg["meta_data"], index=False)

print(f"Saved {len(meta_rows)} chunks (errors: {len(errors)})")
if errors:
    print("Sample errors:", errors[:5])

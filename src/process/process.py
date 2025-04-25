#!/usr/bin/env python3
"""
process_gold.py – Generate high-confidence “golden” training chunks for BirdCLEF-2025.

Pipeline
--------
1. Read **process.yaml** for paths and hyper-parameters.
2. Filter `train.csv` to recordings that satisfy:
   • rating ≥ selection.golden_rating (default 5)
   • zero secondary labels when selection.require_single_label is true.
3. For each qualifying recording:
   a. Deduplicate by MD5 hash of the raw waveform.
   b. Trim leading/trailing silence.
   c. Ensure ≥ audio.min_duration by cyclic-padding.
   d. Slide a 10-second window with 5-second hop:
      – Skip if too silent or contains human speech.
      – Save mel-spectrogram and 1-hot label vector.
      – Record metadata with weight = `labeling.golden_label_weight`.
4. Persist updated `audio_hashes.txt` and `train_metadata.csv`.
"""
from __future__ import annotations
import hashlib
import logging
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils import utils
from src.inference import pseudo_label_inference

with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
audio_cfg = CFG["audio"]
chunk_cfg = CFG["chunking"]
mel_cfg = CFG["mel"]
sel_cfg = CFG["selection"]
label_cfg = CFG["labeling"]
dedup_cfg = CFG["deduplication"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_gold")

AUDIO_DIR = Path(paths_cfg["audio_dir"])
PROCESSED_DIR = Path(paths_cfg["processed_dir"])
MEL_DIR = PROCESSED_DIR / "mels"
LABEL_DIR = PROCESSED_DIR / "labels"
MEL_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = Path(paths_cfg["train_metadata"])
TRAIN_CSV = Path(paths_cfg["train_csv"])
HASH_FILE = PROCESSED_DIR / "audio_hashes.txt"

class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), TRAIN_CSV)
NUM_CLASSES = len(class_list)
seen_hashes: set[str] = set()
if HASH_FILE.exists():
    seen_hashes.update(HASH_FILE.read_text().splitlines())

df = pd.read_csv(TRAIN_CSV)

minimum_rating = sel_cfg["minimum_rating"]
max_count = sel_cfg["max_count"]
rare_species_thresh = sel_cfg["rare_species_threshold"]
primary_label_counts = df.groupby("primary_label")["filename"].transform("count")

minority = df[primary_label_counts < rare_species_thresh]
good_rating = df[df["rating"] >= minimum_rating]
not_too_much = df[primary_label_counts < max_count]

base_df = pd.concat([minority, good_rating, not_too_much], ignore_index=True)
sampled_df = pd.DataFrame()

label_count_base_df = base_df.groupby("primary_label")["filename"].transform("count").proportions()
for label in label_count_base_df.index:
    if label_count_base_df[label] > 0.2:
        sample that label from base df to reduce the number of samples in that label
        save that change into base df
        After this, base df would have all 206 label classes, each not exceeding 20% percent proportion based on primary label.
    Then, we sample from base_df each label class. outcome I want is base df with all minority and sampled metadata from other classes.
    This sampled_df must include all 206 label classes, each not exceeding 20% percent proportion based on primary label.

dedup_df = sampled_df.drop_duplicates(subset=["filename"])
chunk_sec = chunk_cfg["train_chunk_duration"]
hop_sec = chunk_cfg["train_chunk_hop"]
sample_rate = audio_cfg["sample_rate"]
chunk_samples = int(chunk_sec * sample_rate)
hop_samples = int(hop_sec * sample_rate)

meta_rows: List[dict] = []
for rec in sampled_df.itertuples(index=False):
    fname = rec.filename
    label = str(rec.primary_label)
    audio_path = AUDIO_DIR / fname
    if not audio_path.exists():
        log.warning("Missing file: %s", fname)
        continue
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    h = hashlib.md5(y.tobytes()).hexdigest()
    if dedup_cfg.get("enabled", False) and h in seen_hashes:
        continue
    seen_hashes.add(h)
    if audio_cfg.get("trim_top_db") is not None:
        y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    if y.size == 0:
        continue
    min_dur = int(audio_cfg["min_duration"] * sample_rate)
    if y.size < min_dur:
        reps = int(np.ceil(min_dur / y.size))
        y = np.tile(y, reps)[:min_dur]
    total = len(y)
    ptr = 0
    while ptr + chunk_samples <= total:
        chunk = y[ptr : ptr + chunk_samples]
        ptr += hop_samples
        if utils.is_silent(chunk, db_thresh=audio_cfg.get("silence_thresh_db", -50.0)):
            continue
        if utils.contains_voice(chunk, sample_rate):
            continue
        m = librosa.feature.melspectrogram(
            y=chunk,
            sr=sample_rate,
            n_fft=mel_cfg["n_fft"],
            hop_length=mel_cfg["hop_length"],
            n_mels=mel_cfg["n_mels"],
            fmin=mel_cfg["fmin"],
            fmax=mel_cfg["fmax"],
            power=mel_cfg["power"],
        )
        mel_db = librosa.power_to_db(m, ref=np.max)
        mel_db = utils.resize_mel(mel_db, *mel_cfg["target_shape"]).astype(np.float32)
        chunk_id = utils.hash_chunk_id(fname, ptr / sample_rate)
        mel_path = MEL_DIR / f"{chunk_id}.npy"
        label_path = LABEL_DIR / f"{chunk_id}.npy"
        np.save(mel_path, mel_db)
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        idx = class_map.get(label)
        if idx is None:
            continue
        lbl[idx] = 1.0
        np.save(label_path, lbl)
        meta_rows.append(
            {
                "filename": fname,
                "end_sec": round(ptr / sample_rate, 3),
                "mel_path": str(mel_path),
                "label_path": str(label_path),
                "weight": float(label_cfg.get("golden_label_weight", 1.0)),
            }
        )

meta_df = pd.DataFrame(meta_rows)

# Unlabled Soundscapes Processing


if METADATA_CSV.exists():
    existing = pd.read_csv(METADATA_CSV)
    meta_df = pd.concat([existing, meta_df], ignore_index=True)
meta_df.to_csv(METADATA_CSV, index=False)

with HASH_FILE.open("w") as f:
    f.write("\n".join(sorted(seen_hashes)))

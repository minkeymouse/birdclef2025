#!/usr/bin/env python3
"""
process_gold.py – Generate high‑confidence “golden” training chunks for BirdCLEF‑2025.

Pipeline
--------
1. Read **process.yaml** for paths and hyper‑parameters.
2. Filter `train.csv` to recordings that satisfy:
   • rating ≥ selection.golden_rating (default 5)
   • zero secondary labels when selection.require_single_label is true.
3. For each qualifying recording
   a. Deduplicate by MD5 hash of the resampled raw waveform.
   b. Trim leading/trailing silence (librosa.effects.trim).
   c. Ensure ≥ audio.min_duration by cyclic‑padding.
   d. Slide a 10‑second window with 5‑second hop:
      – Skip the chunk if it is too silent (mean power < −50 dB).
      – Skip the chunk if Silero VAD finds human speech.
   e. Convert the chunk to a mel‑spectrogram (dB) and resize to mel.target_shape.
   f. Save
      • `mels/<chunk_id>.npy` – mel array (float32)
      • `labels/<chunk_id>.npy` – 206‑dim 1‑hot label vector
      • Append metadata row: filename, end_sec, mel_path, label_path, weight=1.0.
4. Persist updated audio_hashes.txt so later stages can skip duplicates.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
import yaml
from PIL import Image

import utils  # project‑local helper module providing taxonomy + hashing utilities

# ------------------------------------------------------------------------------
# Configuration & logging
# ------------------------------------------------------------------------------
with open("process.yaml", "r", encoding="utf-8") as f:
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

# ------------------------------------------------------------------------------
# Directory setup
# ------------------------------------------------------------------------------
AUDIO_DIR = Path(paths_cfg["audio_dir"])
PROCESSED_DIR = Path(paths_cfg["processed_dir"])
MEL_DIR = PROCESSED_DIR / "mels"
LABEL_DIR = PROCESSED_DIR / "labels"
for d in (MEL_DIR, LABEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

METADATA_CSV = Path(paths_cfg["train_metadata"])
TRAIN_CSV = Path(paths_cfg["train_csv"])

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
sample_rate = audio_cfg["sample_rate"]
chunk_samples = int(chunk_cfg["train_chunk_duration"] * sample_rate)
hop_samples = int(chunk_cfg["train_chunk_hop"] * sample_rate)


def load_audio_mono(path: Path) -> np.ndarray:
    """Load audio as float32 mono array at target sample rate."""
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y


def is_silent(y: np.ndarray, db_thresh: float = -50.0) -> bool:
    """Return True if RMS power is below `db_thresh` dBFS."""
    db = 10 * np.log10(np.maximum(1e-12, np.mean(y ** 2)))
    return db < db_thresh


def resize_mel(mel_db: np.ndarray) -> np.ndarray:
    """Resize a mel‑spectrogram to the configured target shape using bilinear interpolation."""
    h, w = mel_db.shape
    target_h, target_w = mel_cfg["target_shape"]
    if (h, w) == (target_h, target_w):
        return mel_db
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    img = Image.fromarray((mel_norm * 255).astype(np.uint8))
    img = img.resize((target_w, target_h), Image.BILINEAR)
    mel_resized = np.array(img).astype(np.float32) / 255.0
    return mel_resized * (mel_db.max() - mel_db.min() + 1e-6) + mel_db.min()

# -------------------- Voice‑activity detection ---------------------------------
try:
    vad_model, vad_utils = utils.load_vad()  # returns (model, helpers dict)
    get_speech_timestamps = vad_utils["get_speech_timestamps"]
except Exception:
    log.warning("VAD model unavailable – skipping human‑voice filtering.")
    vad_model = None
    get_speech_timestamps = None


def contains_voice(chunk: np.ndarray) -> bool:
    """Detect human speech in a chunk using Silero VAD."""
    if vad_model is None:
        return False
    ts = get_speech_timestamps(chunk, vad_model, threshold=0.5, sampling_rate=sample_rate)
    return bool(ts)

# ------------------------------------------------------------------------------
# Class mapping (species‑code → index)
# ------------------------------------------------------------------------------
class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), TRAIN_CSV)
NUM_CLASSES = len(class_list)
log.info("Loaded %d target classes.", NUM_CLASSES)

# ------------------------------------------------------------------------------
# Deduplication cache
# ------------------------------------------------------------------------------
HASH_FILE = PROCESSED_DIR / "audio_hashes.txt"
seen_hashes: set[str] = set()
if HASH_FILE.exists():
    seen_hashes.update(h.strip() for h in HASH_FILE.read_text().splitlines())

# ------------------------------------------------------------------------------
# Filter train.csv → golden recordings
# ------------------------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)

golden_df = df[df["rating"] >= sel_cfg["golden_rating"]]
if sel_cfg.get("require_single_label", True):
    if "secondary_labels" in golden_df.columns:
        golden_df = golden_df[
            golden_df["secondary_labels"].isna()
            | (golden_df["secondary_labels"].astype(str).str.strip() == "[]")
        ]
log.info("Golden recordings selected: %d", len(golden_df))

# ------------------------------------------------------------------------------
# Main processing loop
# ------------------------------------------------------------------------------
meta_rows: List[dict] = []

for row in golden_df.itertuples(index=False):
    rec_filename: str = getattr(row, "filename")
    primary_label = getattr(row, "primary_label")

    audio_path = AUDIO_DIR / str(primary_label) / rec_filename
    if not audio_path.exists():
        # try adding common extensions if missing
        for ext in (".ogg", ".mp3", ".wav"):
            alt = audio_path.with_suffix(ext)
            if alt.exists():
                audio_path = alt
                break
    if not audio_path.exists():
        log.warning("Missing file: %s", rec_filename)
        continue

    # -------- Deduplication check --------
    y = load_audio_mono(audio_path)
    audio_hash = hashlib.md5(y.tobytes()).hexdigest()
    if dedup_cfg["enabled"] and audio_hash in seen_hashes:
        log.debug("Duplicate skip: %s", rec_filename)
        continue
    seen_hashes.add(audio_hash)

    # -------- Silence trimming + min‑length padding --------
    if audio_cfg["trim_top_db"] is not None:
        y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    if len(y) == 0:
        continue
    if len(y) < int(audio_cfg["min_duration"] * sample_rate):
        reps = int(np.ceil(audio_cfg["min_duration"] * sample_rate / len(y)))
        y = np.tile(y, reps)[: int(audio_cfg["min_duration"] * sample_rate)]

    total_samples = len(y)
    ptr = 0
    while ptr < total_samples:
        seg = y[ptr : ptr + chunk_samples]
        if len(seg) < chunk_samples:
            # pad first (and only) short recording; otherwise break
            if ptr == 0:
                reps = int(np.ceil(chunk_samples / len(seg)))
                seg = np.tile(seg, reps)[:chunk_samples]
            else:
                break

        # -------- Quality filters --------
        if is_silent(seg):
            ptr += hop_samples
            continue
        if contains_voice(seg):
            ptr += hop_samples
            continue

        # -------- Mel extraction --------
        mel = librosa.feature.melspectrogram(
            seg,
            sr=sample_rate,
            n_fft=mel_cfg["n_fft"],
            hop_length=mel_cfg["hop_length"],
            n_mels=mel_cfg["n_mels"],
            fmin=mel_cfg["fmin"],
            fmax=mel_cfg["fmax"],
            power=mel_cfg["power"],
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = resize_mel(mel_db).astype(np.float32)

        # -------- Paths & IDs --------
        chunk_id = utils.hash_chunk_id(rec_filename, ptr / sample_rate)
        mel_path = MEL_DIR / f"{chunk_id}.npy"
        label_path = LABEL_DIR / f"{chunk_id}.npy"

        # -------- Label vector --------
        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        if primary_label in class_map:
            label_vec[class_map[primary_label]] = 1.0
        else:
            log.warning("Unknown species code: %s", primary_label)
            ptr += hop_samples
            continue

        # save arrays
        np.save(mel_path, mel_db)
        np.save(label_path, label_vec)

        # metadata row uses end‑time of the chunk
        meta_rows.append(
            {
                "filename": rec_filename,
                "end_sec": round((ptr + chunk_samples) / sample_rate, 3),
                "mel_path": str(mel_path),
                "label_path": str(label_path),
                "weight": 1.0,
            }
        )

        ptr += hop_samples

    log.info("Processed %s", rec_filename)

# ------------------------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------------------------
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(METADATA_CSV, index=False)
log.info("Saved %d golden chunks → %s", len(meta_df), METADATA_CSV)

if dedup_cfg["enabled"]:
    HASH_FILE.write_text("\n".join(seen_hashes))
    log.info("Updated hash cache (%d entries).", len(seen_hashes))

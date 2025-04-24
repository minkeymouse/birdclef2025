#!/usr/bin/env python3
"""
process_pseudo.py – Expand training data with high‑confidence pseudo‑labels.

After the initial (golden/rare) model training round, this script scans the
whole *train_audio* archive again and adds chunks **not yet represented** in
`train_metadata.csv`.

For every unseen recording it:
1. Runs VAD + silence checks to discard human speech and empty segments.
2. Splits into the same 10‑s / 5‑s‑hop windows used for training.
3. Predicts class probabilities with all TorchScript checkpoints in the models directory.
4. Accepts a chunk when `max(probabilities) ≥ selection.pseudo_confidence_threshold`.
5. Saves the mel array + soft‑label vector and appends a metadata row with
   reduced sample weight (`labeling.pseudo_label_weight`).

The script is idempotent – duplicate raw waveforms are skipped via an MD5 hash
cache shared with other stages.
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
import torch
import yaml

import utils  # project‑local helpers (taxonomy, resize_mel, VAD, hash_chunk_id)

# -----------------------------------------------------------------------------
# Configuration & logging
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "process.yaml"
MODELS_DIR = ROOT_DIR / "models"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

paths_cfg = CFG["paths"]
audio_cfg = CFG["audio"]
chunk_cfg = CFG["chunking"]
mel_cfg = CFG["mel"]
sel_cfg = CFG["selection"]
label_cfg = CFG["labeling"]
dedup_cfg = CFG["deduplication"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_pseudo")

# -----------------------------------------------------------------------------
# Paths & directories
# -----------------------------------------------------------------------------
AUDIO_DIR = Path(paths_cfg["audio_dir"]).expanduser()
PROCESSED_DIR = Path(paths_cfg["processed_dir"]).expanduser()
MEL_DIR = PROCESSED_DIR / "mels"
LABEL_DIR = PROCESSED_DIR / "labels"
for d in (MEL_DIR, LABEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path(paths_cfg["train_csv"]).expanduser()
METADATA_CSV = Path(paths_cfg["train_metadata"]).expanduser()
# Derive models directory as sibling of processed data\ nMODELS_DIR = PROCESSED_DIR.parent / "models"

# -----------------------------------------------------------------------------
# Taxonomy & state
# -----------------------------------------------------------------------------
class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), TRAIN_CSV)
NUM_CLASSES = len(class_list)
log.info("Loaded %d classes.", NUM_CLASSES)

sample_rate = audio_cfg["sample_rate"]
chunk_samples = int(chunk_cfg["train_chunk_duration"] * sample_rate)
hop_samples = int(chunk_cfg["train_chunk_hop"] * sample_rate)

# -----------------------------------------------------------------------------
# Deduplication cache (shared)
# -----------------------------------------------------------------------------
hash_file = PROCESSED_DIR / "audio_hashes.txt"
seen_hashes: set[str] = set(hash_file.read_text().split()) if hash_file.exists() else set()

# -----------------------------------------------------------------------------
# Previously seen recordings (any chunk) – skip those files entirely
# -----------------------------------------------------------------------------
if METADATA_CSV.exists():
    used_files = set(pd.read_csv(METADATA_CSV, usecols=["filename"]).filename.astype(str))
else:
    used_files = set()

# -----------------------------------------------------------------------------
# Voice‑activity detection helper
# -----------------------------------------------------------------------------
try:
    vad_model, vad_utils = utils.load_vad()
    get_speech_timestamps = vad_utils["get_speech_timestamps"]
except Exception:
    log.warning("VAD not available – speech filtering disabled.")
    vad_model = None
    get_speech_timestamps = None


def contains_voice(samples: np.ndarray) -> bool:
    if vad_model is None:
        return False
    ts = get_speech_timestamps(samples, vad_model, sampling_rate=sample_rate, threshold=0.5)
    return bool(ts)

# -----------------------------------------------------------------------------
# Load ensemble of TorchScript models
# -----------------------------------------------------------------------------
ensemble: List[torch.jit.ScriptModule] = []
for ckpt in MODELS_DIR.glob("**/*.ts.pt"):
    try:
        model = torch.jit.load(str(ckpt), map_location="cuda")
        model.eval()
        ensemble.append(model)
        log.info("Loaded model: %s", ckpt.name)
    except Exception as e:
        log.warning("Skipping %s (load error: %s)", ckpt.name, e)

if not ensemble:
    log.error("No TorchScript checkpoints found under %s – aborting.", MODELS_DIR)
    raise SystemExit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for m in ensemble:
    m.to(DEVICE)

# -----------------------------------------------------------------------------
# Silence filter
# -----------------------------------------------------------------------------
def is_silent(wave: np.ndarray, thresh_db: float = -50.0) -> bool:
    db = 10 * np.log10(np.maximum(1e-12, np.mean(wave**2)))
    return db < thresh_db

# -----------------------------------------------------------------------------
# Main pseudo‑labeling loop
# -----------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
meta_rows: List[dict] = []

for rec in train_df.itertuples(index=False):
    rec_file = str(rec.filename)
    primary_label = rec.primary_label

    # skip recordings already represented
    if rec_file in used_files:
        continue

    wav_path = AUDIO_DIR / str(primary_label) / rec_file
    if not wav_path.exists():
        for ext in (".ogg", ".mp3", ".wav"):
            alt = wav_path.with_suffix(ext)
            if alt.exists():
                wav_path = alt
                break
    if not wav_path.exists():
        log.debug("Missing file: %s", rec_file)
        continue

    # -------- Deduplicate raw waveform --------
    y, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    h = hashlib.md5(y.tobytes()).hexdigest()
    if dedup_cfg["enabled"] and h in seen_hashes:
        continue
    seen_hashes.add(h)

    # -------- Pre‑cleaning (trim silence) --------
    if audio_cfg.get("trim_top_db") is not None:
        y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    if len(y) == 0:
        continue

    # pad to match hop
    pad = (hop_samples - len(y) % hop_samples) % hop_samples
    if pad:
        y = np.pad(y, (0, pad))
    total_len = len(y)

    ptr = 0
    while ptr + chunk_samples <= total_len:
        chunk = y[ptr:ptr + chunk_samples]
        ptr += hop_samples

        if is_silent(chunk):
            continue
        if contains_voice(chunk):
            continue

        # -------- Mel‑spectrogram + resize --------
        mel = librosa.feature.melspectrogram(
            chunk, sr=sample_rate,
            n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
            n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"],
            power=mel_cfg["power"],
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = utils.resize_mel(mel_db, *mel_cfg["target_shape"]).astype(np.float32)

        # -------- Inference --------
        x = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(DEVICE)
        probs_sum = np.zeros(NUM_CLASSES, dtype=np.float32)
        with torch.no_grad():
            for model in ensemble:
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                probs_sum += probs.astype(np.float32)
        probs_avg = probs_sum / len(ensemble)
        if probs_avg.max() < sel_cfg["pseudo_confidence_threshold"]:
            continue

        # -------- Persist --------
        chunk_id = utils.hash_chunk_id(rec_file, ptr / sample_rate)
        mel_path = MEL_DIR / f"{chunk_id}.npy"
        label_path = LABEL_DIR / f"{chunk_id}.npy"
        np.save(mel_path, mel_db)
        np.save(label_path, probs_avg.astype(np.float32))

        meta_rows.append({
            "filename": rec_file,
            "end_sec": round((ptr) / sample_rate, 3),
            "mel_path": str(mel_path),
            "label_path": str(label_path),
            "weight": float(label_cfg["pseudo_label_weight"]),
        })

    log.info("Pseudo‑processed %s", rec_file)

# -----------------------------------------------------------------------------
# Save metadata & hash cache
# -----------------------------------------------------------------------------
if meta_rows:
    out_df = pd.DataFrame(meta_rows)
    header = not METADATA_CSV.exists()
    out_df.to_csv(METADATA_CSV, mode="a", index=False, header=header)
    log.info("Added %d pseudo‑labelled chunks → %s", len(out_df), METADATA_CSV)
else:
    log.info("No new pseudo‑labelled chunks generated.")

if dedup_cfg.get("enabled", False):
    hash_file.write_text("\n".join(sorted(seen_hashes)))

#!/usr/bin/env python3
"""
process_gold.py – Create golden training set from 5-star single-label recordings.

Filters the master training CSV for high-quality recordings (rating >= 5 with no secondary species),
then for each selected recording:
 - Loads audio, trims silence, deduplicates identical audio,
 - Splits into chunks of fixed duration (with overlap), padding if needed for short audio,
 - Extracts mel spectrogram for each chunk and saves it,
 - Creates one-hot label vector for the primary species,
 - Records metadata for each chunk in train_metadata.csv (weight=1, pseudo=0).

Run this script before initial training.
"""
import os, ast, hashlib, logging
from pathlib import Path
import yaml
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import utils  # import shared utility functions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_gold")

# Load configuration
with open("process.yaml", "r") as f:
    config = yaml.safe_load(f)
paths_cfg = config["paths"]
audio_cfg = config["audio"]
chunk_cfg = config["chunking"]
mel_cfg = config["mel"]
sel_cfg = config["selection"]
label_cfg = config["labeling"]
# Define important parameters
DATA_ROOT = Path(paths_cfg["data_root"])
AUDIO_DIR = Path(paths_cfg["audio_dir"])
PROCESSED_DIR = Path(paths_cfg["processed_dir"])
TRAIN_CSV = Path(paths_cfg["train_csv"])
METADATA_CSV = Path(paths_cfg["train_metadata"])
DEDUP_ENABLED = config["deduplication"]["enabled"]
# Create output directories
mel_dir = PROCESSED_DIR / "mels"
label_dir = PROCESSED_DIR / "labels"
mel_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

# Load taxonomy to get class index mapping
class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), TRAIN_CSV)
num_classes = len(class_list)
logger.info(f"Loaded {num_classes} target classes.")

# Prepare to track duplicates
seen_hashes = set()
hash_file = PROCESSED_DIR / "audio_hashes.txt"
if hash_file.exists():
    # Load existing hashes (if any previous processing was done)
    with open(hash_file, 'r') as hf:
        for line in hf:
            seen_hashes.add(line.strip())

# Read training metadata CSV
train_df = pd.read_csv(TRAIN_CSV)
logger.info(f"Total recordings in train.csv: {len(train_df)}")
# Filter for golden criteria
golden_df = train_df[ (train_df["rating"] >= sel_cfg["golden_rating"]) ]
if sel_cfg.get("require_single_label", True):
    # Exclude any recording that has secondary labels listed
    if "secondary_labels" in golden_df.columns:
        # Some secondary_labels might be NaN or empty string if none
        golden_df = golden_df[golden_df["secondary_labels"].isna() | (golden_df["secondary_labels"] == "[]")]
logger.info(f"Golden set candidate recordings: {len(golden_df)}")

metadata_entries = []  # will collect rows for train_metadata
for _, row in golden_df.iterrows():
    file_id = row.get("filename") or row.get("file") or row.get("recording_id")
    primary_label = row["primary_label"]
    rating = row.get("rating", 5)
    # Only proceed if meets minimum rating (should already, but just in case)
    if rating < sel_cfg["golden_rating"]:
        continue
    # Build audio file path
    audio_path = AUDIO_DIR / primary_label / str(file_id)
    if not audio_path.exists():
        # Try adding extension if missing
        # Common extensions in dataset might be .ogg or .mp3
        if not str(audio_path).endswith(('.wav','.ogg','.mp3')):
            for ext in [".ogg", ".mp3", ".wav"]:
                if (AUDIO_DIR / primary_label / f"{file_id}{ext}").exists():
                    audio_path = AUDIO_DIR / primary_label / f"{file_id}{ext}"
                    break
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        continue

    # Deduplication: check audio hash to skip identical files
    y, sr = librosa.load(audio_path, sr=audio_cfg["sample_rate"], mono=True)
    # Compute a hash of raw audio data (after resample) for dedup
    audio_hash = hashlib.md5(y.tobytes()).hexdigest()
    if DEDUP_ENABLED and audio_hash in seen_hashes:
        logger.info(f"Skipping duplicate audio: {file_id} ({primary_label})")
        continue
    seen_hashes.add(audio_hash)

    # Trim leading/trailing silence
    if audio_cfg["trim_top_db"] is not None:
        y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    # Ensure minimum duration by padding or looping
    chunk_dur = int(chunk_cfg["train_chunk_duration"] * audio_cfg["sample_rate"])
    if len(y) == 0:
        continue  # skip if completely silent
    if len(y) < int(audio_cfg["min_duration"] * audio_cfg["sample_rate"]):
        # If shorter than minimum duration, loop it to reach that length
        repeat_count = int(np.ceil((audio_cfg["min_duration"] * audio_cfg["sample_rate"]) / len(y)))
        y = np.tile(y, repeat_count)[:int(audio_cfg["min_duration"] * audio_cfg["sample_rate"])]

    # Segment into overlapping chunks
    hop_samples = int(chunk_cfg["train_chunk_hop"] * audio_cfg["sample_rate"])
    total_samples = len(y)
    start = 0
    while start < total_samples:
        end = start + chunk_dur
        if end > total_samples:
            # If this is the last segment and shorter than chunk_dur, decide whether to include
            if start == 0:
                # If the whole recording is shorter than chunk duration, pad it cyclically
                if len(y) < chunk_dur:
                    repeats = (chunk_dur // len(y)) + 1
                    y_pad = np.tile(y, repeats)[:chunk_dur]
                else:
                    y_pad = y
                segment = y_pad
            else:
                # Break out if last segment would be incomplete (already covered in previous segment overlap)
                break
        else:
            segment = y[start:end]
        # Calculate chunk start time in seconds
        start_sec = start / audio_cfg["sample_rate"]
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(segment, sr=audio_cfg["sample_rate"],
                                                 n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
                                                 n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"], power=mel_cfg["power"])
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Resize mel spectrogram to target shape
        mel_height, mel_width = mel_db.shape
        target_h, target_w = mel_cfg["target_shape"]
        if (mel_height, mel_width) != (target_h, target_w):
            # Normalize mel values to [0,255] for image resizing
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            mel_img = (mel_norm * 255).astype(np.uint8)
            mel_img = Image.fromarray(mel_img)
            mel_img = mel_img.resize((target_w, target_h), Image.BILINEAR)
            mel_resized = np.array(mel_img).astype(np.float32) / 255.0
            # Convert back to dB scale range
            mel_resized = mel_resized * (mel_db.max() - mel_db.min() + 1e-6) + mel_db.min()
            mel_db = mel_resized
        # Save mel spectrogram array
        chunk_id = utils.hash_chunk_id(str(file_id), start_sec)
        mel_path = mel_dir / f"{chunk_id}.npy"
        label_path = label_dir / f"{chunk_id}.npy"
        np.save(mel_path, mel_db.astype(np.float32))
        # Create label vector (one-hot since no secondary labels)
        label_vec = np.zeros(num_classes, dtype=np.float32)
        label_index = class_map.get(primary_label)
        if label_index is None:
            # If species not in our class map (should not happen if taxonomy covers all)
            logger.warning(f"Species {primary_label} not in class map, skipping chunk.")
            start += hop_samples
            continue
        label_vec[label_index] = 1.0  # one-hot for primary species
        np.save(label_path, label_vec)
        # Add metadata entry
        metadata_entries.append({
            "filename": str(file_id),
            "start_sec": round(start_sec, 3),
            "mel_path": str(mel_path),
            "label_path": str(label_path),
            "weight": 1.0,            # golden samples weight=1
            "pseudo": 0               # human-labeled
        })
        start += hop_samples
    logger.info(f"Processed golden file: {file_id} ({primary_label})")

# Save metadata to CSV
md_df = pd.DataFrame(metadata_entries, columns=["filename","start_sec","mel_path","label_path","weight","pseudo"])
md_df.to_csv(METADATA_CSV, index=False)
logger.info(f"Golden set processing complete. Chunks saved: {len(md_df)}. Metadata written to {METADATA_CSV}")

# Save updated hash list for deduplication reference
if DEDUP_ENABLED:
    with open(hash_file, 'w') as hf:
        for h in seen_hashes:
            hf.write(f"{h}\n")

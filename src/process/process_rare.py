#!/usr/bin/env python3
"""
process_rare.py – Add rare classes to training set by including lower-rated recordings.

Identifies bird species with few available recordings (based on train.csv and a threshold).
For each rare species, includes all recordings of that species (down to min_rating) that were not in the golden set.
Processes each new recording similar to process_gold (trimming, chunking, mel extraction).
Also incorporates synthetic audio for those species if available (from synthetic_dir).
Appends the new chunks to the existing train_metadata.csv with appropriate labels and weights.
Run this after process_gold (and initial training, if desired) to enrich the training data with minority classes.
"""
import os, ast, hashlib, logging
from pathlib import Path
import yaml
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_rare")

# Load configuration
with open("process.yaml", "r") as f:
    config = yaml.safe_load(f)
paths_cfg = config["paths"]
audio_cfg = config["audio"]
chunk_cfg = config["chunking"]
mel_cfg = config["mel"]
sel_cfg = config["selection"]
label_cfg = config["labeling"]
DATA_ROOT = Path(paths_cfg["data_root"])
AUDIO_DIR = Path(paths_cfg["audio_dir"])
SYN_DIR = Path(paths_cfg.get("synthetic_dir", ""))
PROCESSED_DIR = Path(paths_cfg["processed_dir"])
TRAIN_CSV = Path(paths_cfg["train_csv"])
METADATA_CSV = Path(paths_cfg["train_metadata"])
DEDUP_ENABLED = config["deduplication"]["enabled"]

mel_dir = PROCESSED_DIR / "mels"
label_dir = PROCESSED_DIR / "labels"
mel_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

# Load class mapping
class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), TRAIN_CSV)
num_classes = len(class_list)

# Prepare deduplication set
seen_hashes = set()
hash_file = PROCESSED_DIR / "audio_hashes.txt"
if hash_file.exists():
    with open(hash_file, 'r') as hf:
        for line in hf:
            seen_hashes.add(line.strip())

# Determine rare species based on occurrence counts
train_df = pd.read_csv(TRAIN_CSV)
# Count total recordings per species (all ratings)
species_counts = train_df.groupby("primary_label")["filename"].count().to_dict()
rare_species = [sp for sp, count in species_counts.items() if count < sel_cfg["rare_species_threshold"]]
logger.info(f"Identified {len(rare_species)} rare species with < {sel_cfg['rare_species_threshold']} recordings.")

# Also ensure we only add species that were not fully covered in golden
# Actually, if a rare species had some golden entries, we may still add more of its recordings
# So no need to exclude those, we will skip duplicates by file name or hash anyway.
# Load existing train_metadata to skip files already processed
existing_md = pd.read_csv(METADATA_CSV)
used_files = set(existing_md["filename"].astype(str).unique())

new_entries = []
# Process real recordings for rare species
for species in rare_species:
    # Find all recordings of this species in train_df
    species_df = train_df[train_df["primary_label"] == species]
    if species_df.empty:
        continue
    for _, row in species_df.iterrows():
        file_id = row.get("filename") or row.get("file") or row.get("recording_id")
        if str(file_id) in used_files:
            # Already included in golden set
            continue
        rating = row.get("rating", 0)
        if rating < audio_cfg["min_rating"]:
            continue
        secondary_labels = []
        if "secondary_labels" in row and isinstance(row["secondary_labels"], str) and row["secondary_labels"].strip():
            # Parse secondary labels list from string
            try:
                secondary_labels = ast.literal_eval(row["secondary_labels"])
            except Exception:
                secondary_labels = []
        audio_path = AUDIO_DIR / species / str(file_id)
        if not audio_path.exists():
            if not str(audio_path).endswith(('.wav','.ogg','.mp3')):
                for ext in [".ogg", ".mp3", ".wav"]:
                    if (AUDIO_DIR / species / f"{file_id}{ext}").exists():
                        audio_path = AUDIO_DIR / species / f"{file_id}{ext}"
                        break
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        # Deduplication check
        y, sr = librosa.load(audio_path, sr=audio_cfg["sample_rate"], mono=True)
        audio_hash = hashlib.md5(y.tobytes()).hexdigest()
        if DEDUP_ENABLED and audio_hash in seen_hashes:
            logger.info(f"Skipping duplicate audio: {file_id} ({species})")
            continue
        seen_hashes.add(audio_hash)

        # Trim silence from ends
        if audio_cfg["trim_top_db"] is not None:
            y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
        if len(y) == 0:
            continue
        # Pad short recordings to min_duration if needed
        if len(y) < int(audio_cfg["min_duration"] * audio_cfg["sample_rate"]):
            repeats = int(np.ceil((audio_cfg["min_duration"] * audio_cfg["sample_rate"]) / len(y)))
            y = np.tile(y, repeats)[:int(audio_cfg["min_duration"] * audio_cfg["sample_rate"])]

        # Chunk into segments
        chunk_samples = int(chunk_cfg["train_chunk_duration"] * audio_cfg["sample_rate"])
        hop_samples = int(chunk_cfg["train_chunk_hop"] * audio_cfg["sample_rate"])
        total_samples = len(y)
        start = 0
        while start < total_samples:
            end = start + chunk_samples
            if end > total_samples:
                if start == 0:
                    # file shorter than one chunk
                    if len(y) < chunk_samples:
                        repeats = (chunk_samples // len(y)) + 1
                        seg = np.tile(y, repeats)[:chunk_samples]
                    else:
                        seg = y
                else:
                    break
            else:
                seg = y[start:end]
            start_sec = start / audio_cfg["sample_rate"]
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(seg, sr=audio_cfg["sample_rate"],
                                                     n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
                                                     n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"], power=mel_cfg["power"])
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            # Resize mel to target shape
            mel_height, mel_width = mel_db.shape
            target_h, target_w = mel_cfg["target_shape"]
            if (mel_height, mel_width) != (target_h, target_w):
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
                mel_img = Image.fromarray((mel_norm*255).astype(np.uint8))
                mel_img = mel_img.resize((target_w, target_h), Image.BILINEAR)
                mel_resized = np.array(mel_img).astype(np.float32) / 255.0
                mel_resized = mel_resized * (mel_db.max() - mel_db.min() + 1e-6) + mel_db.min()
                mel_db = mel_resized
            # Save mel
            chunk_id = utils.hash_chunk_id(str(file_id), start_sec)
            mel_path = mel_dir / f"{chunk_id}.npy"
            label_path = label_dir / f"{chunk_id}.npy"
            np.save(mel_path, mel_db.astype(np.float32))
            # Create label vector (may be multi-label if secondary species present)
            prim_label = species
            sec_labels = secondary_labels if label_cfg["use_soft_labels"] and secondary_labels else []
            label_vec = utils.create_label_vector(prim_label, sec_labels, class_map,
                                                  primary_weight=label_cfg["primary_label_weight"],
                                                  secondary_weight=label_cfg["secondary_label_weight"],
                                                  use_soft=label_cfg["use_soft_labels"])
            np.save(label_path, label_vec.astype(np.float32))
            # weight for rare class sample
            w = label_cfg.get("rare_label_weight", 1.0)
            new_entries.append({
                "filename": str(file_id),
                "start_sec": round(start_sec, 3),
                "mel_path": str(mel_path),
                "label_path": str(label_path),
                "weight": float(w),
                "pseudo": 0
            })
            start += hop_samples
        logger.info(f"Processed rare file: {file_id} (species {species})")

# Process synthetic audio for rare species (if any synthetic files exist)
if SYN_DIR and Path(SYN_DIR).exists():
    syn_files = list(Path(SYN_DIR).rglob("*.*"))
    logger.info(f"Found {len(syn_files)} synthetic audio files.")
    for syn_path in syn_files:
        species = syn_path.parent.name  # assuming synthetic files are in subdirectories named by species
        if species not in rare_species:
            # Only include synthetic for species truly rare or needed
            # (Optionally could include synthetic for others if provided, but skip if not rare)
            continue
        file_id = syn_path.stem
        # Dedup check: synthetic likely unique, but check to avoid duplicates among synthetic
        y, sr = librosa.load(syn_path, sr=audio_cfg["sample_rate"], mono=True)
        audio_hash = hashlib.md5(y.tobytes()).hexdigest()
        if DEDUP_ENABLED and audio_hash in seen_hashes:
            logger.info(f"Skipping duplicate synthetic audio: {syn_path.name}")
            continue
        seen_hashes.add(audio_hash)
        # Trim silence (if any) and pad if needed
        if audio_cfg["trim_top_db"] is not None:
            y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
        if len(y) == 0:
            continue
        if len(y) < int(audio_cfg["min_duration"] * audio_cfg["sample_rate"]):
            repeats = int(np.ceil((audio_cfg["min_duration"] * audio_cfg["sample_rate"]) / len(y)))
            y = np.tile(y, repeats)[:int(audio_cfg["min_duration"] * audio_cfg["sample_rate"])]

        # For synthetic, many are short single calls; we'll create at least one chunk (pad to full chunk_dur)
        chunk_samples = int(chunk_cfg["train_chunk_duration"] * audio_cfg["sample_rate"])
        if len(y) < chunk_samples:
            repeats = (chunk_samples // len(y)) + 1
            y = np.tile(y, repeats)[:chunk_samples]
        total_samples = len(y)
        start = 0
        while start < total_samples:
            end = start + chunk_samples
            if end > total_samples:
                segment = y[start:]  # last segment (should not happen if we padded to chunk length)
            else:
                segment = y[start:end]
            start_sec = start / audio_cfg["sample_rate"]
            mel_spec = librosa.feature.melspectrogram(segment, sr=audio_cfg["sample_rate"],
                                                     n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
                                                     n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"], power=mel_cfg["power"])
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_height, mel_width = mel_db.shape
            target_h, target_w = mel_cfg["target_shape"]
            if (mel_height, mel_width) != (target_h, target_w):
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
                mel_img = Image.fromarray((mel_norm*255).astype(np.uint8))
                mel_img = mel_img.resize((target_w, target_h), Image.BILINEAR)
                mel_resized = np.array(mel_img).astype(np.float32) / 255.0
                mel_resized = mel_resized * (mel_db.max() - mel_db.min() + 1e-6) + mel_db.min()
                mel_db = mel_resized
            chunk_id = utils.hash_chunk_id(syn_path.stem, start_sec)
            mel_path = mel_dir / f"{chunk_id}.npy"
            label_path = label_dir / f"{chunk_id}.npy"
            np.save(mel_path, mel_db.astype(np.float32))
            # Label: one-hot for synthetic (single target species)
            label_vec = np.zeros(num_classes, dtype=np.float32)
            if species in class_map:
                label_vec[class_map[species]] = 1.0
            else:
                logger.warning(f"Synthetic species {species} not in class map, skipping.")
                start += chunk_samples
                continue
            np.save(label_path, label_vec)
            w = label_cfg.get("rare_label_weight", 1.0)  # treat synthetic similar to rare real
            new_entries.append({
                "filename": syn_path.name,
                "start_sec": round(start_sec, 3),
                "mel_path": str(mel_path),
                "label_path": str(label_path),
                "weight": float(w),
                "pseudo": 0
            })
            start += chunk_samples
        logger.info(f"Processed synthetic file: {syn_path.name} (species {species})")

# Append new entries to train_metadata.csv
if new_entries:
    md_df = pd.DataFrame(new_entries, columns=["filename","start_sec","mel_path","label_path","weight","pseudo"])
    md_df.to_csv(METADATA_CSV, mode='a', index=False, header=False)
    logger.info(f"Added {len(md_df)} new entries to train_metadata.csv for rare species.")
else:
    logger.info("No new entries added for rare species.")
# Update hash file
if DEDUP_ENABLED:
    with open(hash_file, 'w') as hf:
        for h in seen_hashes:
            hf.write(f"{h}\n")

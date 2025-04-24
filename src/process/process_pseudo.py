#!/usr/bin/env python3
"""
process_pseudo.py – Generate pseudo-labels for remaining unlabeled recordings.

Uses the initially trained models to predict labels for all recordings not already in the training set.
For each recording not in train_metadata (i.e., not processed in golden or rare stages):
 - Splits the recording into chunks (with overlap) for inference,
 - Averages predictions from all available models (ensemble),
 - For each chunk where the top prediction confidence >= threshold, saves that chunk with the model-provided label probabilities as soft labels,
 - Marks these chunks as pseudo-labeled with a lower weight,
 - Appends the new entries to train_metadata.csv.

Run after initial model training to expand the training dataset with pseudo-labeled data.
"""
import os, hashlib, logging
from pathlib import Path
import yaml
import librosa
import numpy as np
import pandas as pd
import torch
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_pseudo")

# Load config
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
PROCESSED_DIR = Path(paths_cfg["processed_dir"])
TRAIN_CSV = Path(paths_cfg["train_csv"])
METADATA_CSV = Path(paths_cfg["train_metadata"])
models_dir = Path("/data/birdclef/models")  # directory where initial models are stored (from initial_train stage)

# Load class map and list
class_list, class_map = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), TRAIN_CSV)
num_classes = len(class_list)

# Load train_metadata to find used files
if METADATA_CSV.exists():
    used_files = set(pd.read_csv(METADATA_CSV)["filename"].astype(str).unique())
else:
    used_files = set()

# Prepare models for inference (load all initial trained model checkpoints)
# We assume initial models are saved with a known naming scheme (as per initial_train.yaml config)
ensemble_models = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model architecture definitions (match those in initial_train.yaml)
initial_archs = [("efficientnet_b0", 3), ("regnety_008", 3)]
for arch_name, count in initial_archs:
    for i in range(1, count+1):
        ckpt_path = models_dir / f"{arch_name}_initial_{i}.pth"
        if not ckpt_path.exists():
            logger.warning(f"Model checkpoint not found: {ckpt_path}")
            continue
        # Load model (assuming torchscript or state dict saved)
        try:
            model = torch.jit.load(str(ckpt_path)) if ckpt_path.suffix == ".pt" else None
        except Exception:
            model = None
        if model is None:
            # If saved as state dict, you'd need model class definitions to load. 
            # Here we assume model was saved as torchscript or entire model.
            logger.warning(f"Unable to load model (requires definition): {ckpt_path.name}")
            continue
        model.eval().to(device)
        ensemble_models.append(model)
logger.info(f"Loaded {len(ensemble_models)} models for pseudo-label inference.")

if not ensemble_models:
    logger.error("No models loaded for inference. Aborting pseudo-labeling.")
    exit(1)

# Deduplication: prepare seen hashes set including all already used files
seen_hashes = set()
hash_file = PROCESSED_DIR / "audio_hashes.txt"
if hash_file.exists():
    with open(hash_file, 'r') as hf:
        for line in hf:
            seen_hashes.add(line.strip())

new_entries = []
# Iterate over all recordings in train.csv that were not used in training
train_df = pd.read_csv(TRAIN_CSV)
for _, row in train_df.iterrows():
    file_id = str(row.get("filename") or row.get("file") or row.get("recording_id"))
    primary_label = row["primary_label"]
    if file_id in used_files:
        continue
    audio_path = AUDIO_DIR / primary_label / file_id
    if not audio_path.exists():
        # try with extension
        found = False
        for ext in [".ogg", ".mp3", ".wav"]:
            if (AUDIO_DIR / primary_label / f"{file_id}{ext}").exists():
                audio_path = AUDIO_DIR / primary_label / f"{file_id}{ext}"
                found = True
                break
        if not found:
            logger.warning(f"File not found for pseudo-labeling: {file_id}")
            continue

    # Dedup check
    y, sr = librosa.load(audio_path, sr=audio_cfg["sample_rate"], mono=True)
    audio_hash = hashlib.md5(y.tobytes()).hexdigest()
    if config["deduplication"]["enabled"] and audio_hash in seen_hashes:
        logger.info(f"Skipping duplicate (unlabeled) audio: {file_id}")
        continue
    seen_hashes.add(audio_hash)

    # Trim ends silence
    if audio_cfg["trim_top_db"] is not None:
        y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    if len(y) == 0:
        continue

    # Chunk for inference (use same chunk length and hop as training)
    chunk_samples = int(chunk_cfg["train_chunk_duration"] * audio_cfg["sample_rate"])
    hop_samples = int(chunk_cfg["train_chunk_hop"] * audio_cfg["sample_rate"])
    total_samples = len(y)
    # Pad the audio at end to cover last segment fully
    if total_samples < chunk_samples:
        # pad with silence to one chunk length
        pad = chunk_samples - total_samples
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
        total_samples = len(y)
    elif total_samples % hop_samples != 0:
        # pad so that the last start index aligns to cover end
        pad = hop_samples - (total_samples % hop_samples)
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
        total_samples = len(y)

    # Slide through audio
    start = 0
    while start < total_samples:
        end = start + chunk_samples
        if end > total_samples:
            segment = y[start:total_samples]
            # pad this last segment to full length
            segment = np.pad(segment, (0, end - total_samples), mode='constant')
        else:
            segment = y[start:end]
        start_sec = start / audio_cfg["sample_rate"]
        # Compute mel spectrogram for the segment
        mel_spec = librosa.feature.melspectrogram(segment, sr=audio_cfg["sample_rate"],
                                                 n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"],
                                                 n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"], power=mel_cfg["power"])
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Resize mel to target shape
        h, w = mel_db.shape
        target_h, target_w = mel_cfg["target_shape"]
        if (h, w) != (target_h, target_w):
            mel_db = utils.resize_mel(mel_db, target_h, target_w)
        # Convert mel to torch tensor and predict with ensemble
        mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(device)  # shape (1,1,H,W)
        # If model expects 3-channel input, we would stack mel 3 times here. Assume our models were adapted to 1-channel.
        avg_pred = np.zeros(num_classes, dtype=np.float32)
        with torch.no_grad():
            for model in ensemble_models:
                # Each model should output a tensor of shape (1, num_classes) logits
                logits = model(mel_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()  # convert to probabilities
                avg_pred += probs.astype(np.float32)
        avg_pred /= len(ensemble_models)
        # Determine if this chunk has a confident prediction
        top_prob = float(np.max(avg_pred))
        if top_prob >= sel_cfg["pseudo_confidence_threshold"]:
            # Save chunk and pseudo label
            chunk_id = utils.hash_chunk_id(file_id, start_sec)
            mel_path = PROCESSED_DIR / "mels" / f"{chunk_id}.npy"
            label_path = PROCESSED_DIR / "labels" / f"{chunk_id}.npy"
            np.save(mel_path, mel_db.astype(np.float32))
            # Save soft label vector (predicted probabilities)
            label_vec = avg_pred  # already numpy array of float32
            np.save(label_path, label_vec.astype(np.float32))
            # Assign weight (pseudo labels have lower weight)
            w = label_cfg["pseudo_label_weight"]
            # Optionally scale weight by confidence (not explicitly configured, so skipping)
            new_entries.append({
                "filename": file_id,
                "start_sec": round(start_sec, 3),
                "mel_path": str(mel_path),
                "label_path": str(label_path),
                "weight": float(w),
                "pseudo": 1
            })
        start += hop_samples
    logger.info(f"Inferred pseudo-labels for file: {file_id}")

# Append pseudo-labeled entries to metadata
if new_entries:
    md_df = pd.DataFrame(new_entries, columns=["filename","start_sec","mel_path","label_path","weight","pseudo"])
    md_df.to_csv(METADATA_CSV, mode='a', index=False, header=False)
    logger.info(f"Appended {len(md_df)} pseudo-labeled chunks to train_metadata.csv.")
else:
    logger.info("No pseudo-labeled chunks added (no confident predictions).")

# Update dedup hash record
if config["deduplication"]["enabled"]:
    with open(PROCESSED_DIR / "audio_hashes.txt", 'w') as hf:
        for h in seen_hashes:
            hf.write(f"{h}\n")

#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import logging
import sys
from pathlib import Path
from typing import List

import math
import cv2
import time
import os
import librosa
import numpy as np
import pandas as pd
import yaml
import tqdm
import ast

project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils import utils

with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
audio_cfg = CFG["audio"]
sel_cfg = CFG["selection"]
label_cfg = CFG["labeling"]
debug_cfg = CFG["debug"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_gold")

print(f"Debug mode: {'ON' if debug_cfg.enabled else 'OFF'}")
print(f"Max samples to process: {debug_cfg.N_MAX if debug_cfg.N_MAX is not None else 'ALL'}")

print("Loading taxonomy data...")
taxonomy_df = pd.read_csv(f'{paths_cfg.DATA_ROOT}/taxonomy.csv')
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))
print("Loading training metadata...")
train_df = pd.read_csv(f'{paths_cfg.DATA_ROOT}/train.csv')

label_list = sorted(train_df['primary_label'].unique())
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

print(f'Found {len(label_list)} unique species')
working_df = train_df[['primary_label', 'secondary_labels', 'rating', 'filename']].copy()
working_df['primary_label'] = working_df.primary_label.map(label2id)
working_df['secondary_labels'] = working_df.secondary_labels
working_df['filepath'] = working_df.filename
working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
total_samples = min(len(working_df), debug_cfg.N_MAX or len(working_df))

def audio2melspec(audio_data):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.HOP_LENGTH,
        n_mels=audio_cfg.N_MELS,
        fmin=audio_cfg.FMIN,
        fmax=audio_cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def parse_secondary(s):
    if pd.isna(s):
        return []
    elif isinstance(s, str):
        return ast.literal_eval(s)
    else:
        return [s]

print("Starting audio processing...")
print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")
start_time = time.time()

train_duration = audio_cfg.train_duration
train_hop = audio_cfg.train_chunk_hop
meta_rows: List[dict] = []

chunk_samples = int(train_duration * audio_cfg.sample_rate)
hop_samples = int(train_hop * audio_cfg.sample_rate)

errors = []

for i, row in tqdm(working_df.iterrows(), total=total_samples):
    print(f"Processing {i+1}/{total_samples}: {row.filename}")
    if debug_cfg.N_MAX is not None and i >= debug_cfg.N_MAX:
        break

    fname = row.filename
    audio_path = paths_cfg.audio_dir / fname
    secondaries = parse_secondary(row.secondary_labels)
    onehot = np.zeros(len(label_list), dtype=np.float32)
    pid = label2id[row.primary_label]
    onehot[pid] = 1.0

    for sec in secondaries:
        if sec in label2id:
            onehot[label2id[sec]] = 0.3

    rating = row.rating

    if not audio_path.exists():
        log.warning("Missing file: %s", fname)
        continue

    try:
        audio_data, _ = librosa.load(row.filepath, sr=audio_cfg.sample_rate)
        audio_data, _ = librosa.effects.trim(audio_data, top_db=audio_cfg.trim_top_db)
        if audio_data.size == 0:
            continue
        if audio_data.size < audio_cfg.train_duration:
            reps = int(np.ceil(audio_cfg.train_duration / audio_data.size))
            audio_data = np.tile(audio_data, reps)[:audio_cfg.train_duration]
        
        total = len(audio_data)
        ptr = 0
        while ptr + audio_cfg.train_duration <= total:
            chunk = audio_data[ptr : ptr + audio_cfg.train_duration]
            ptr += train_hop
            if utils.is_silent(chunk, db_thresh=audio_cfg.silence_thresh_db):
                continue
            if utils.contains_voice(chunk, audio_cfg.sample_rate):
                continue
            m = audio2melspec(chunk)

            if m.shape != audio_cfg.target_shape:
                m = cv2.resize(m, audio_cfg.target_shape, interpolation=cv2.INTER_LINEAR)
            
            m = m.astype(np.float32)

            chunk_id = utils.hash_chunk_id(fname, ptr / audio_cfg.sample_rate)

            m_path = paths_cfg.mel_dir / f"{chunk_id}.npy"
            label_path = paths_cfg.label_dir / f"{chunk_id}.npy"

            np.save(m_path, m)
            np.save(label_path, onehot)
        
    except Exception as e:
        print(f"Error processing {row.filepath}: {e}")
        errors.append((row.filepath, str(e)))

end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")
print(f"Successfully processed {len(working_df)} files out of {total_samples} total")
print(f"Failed to process {len(errors)} files")








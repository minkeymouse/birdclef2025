#!/usr/bin/env python3
"""
utils.py – Shared utility functions for the BirdCLEF 2025 pipeline.
Includes:
- Taxonomy loading and class mapping,
- Label vector creation for multi-label and pseudo-label samples,
- Chunk ID hashing for unique file naming,
- Mel spectrogram resizing.
"""
import ast
import hashlib
import numpy as np
from PIL import Image
import pandas as pd

def load_taxonomy(taxonomy_csv_path: str, train_csv_path: str):
    """
    Load species list and mapping from taxonomy or train data.
    Returns (class_list, class_map) where class_list is a list of species codes and 
    class_map is a dict mapping species code to index.
    """
    class_list = []
    if taxonomy_csv_path and pd.io.common.file_exists(taxonomy_csv_path):
        # If a taxonomy CSV is provided, use it to get the list of classes (species)
        tax_df = pd.read_csv(taxonomy_csv_path)
        # Assume there's a column with species code (could be 'ebird_code' or 'species_code' or similar)
        for col in ["ebird_code", "species_code", "primary_label"]:
            if col in tax_df.columns:
                class_list = list(pd.unique(tax_df[col]))
                break
    if not class_list:
        # Fallback: derive class list from training data
        train_df = pd.read_csv(train_csv_path)
        class_list = sorted(pd.unique(train_df["primary_label"]))
    # Ensure consistent ordering
    class_list = sorted(class_list)
    class_map = {species: idx for idx, species in enumerate(class_list)}
    return class_list, class_map

def create_label_vector(primary_label: str, secondary_labels: list, class_map: dict,
                        primary_weight: float = 0.7, secondary_weight: float = 0.3,
                        use_soft: bool = True) -> np.ndarray:
    """
    Create a label vector for a training chunk.
    - If use_soft is True and secondary_labels is non-empty, assign primary_label a weight (primary_weight)
      and distribute secondary_weight among secondary_labels (evenly if multiple).
    - If use_soft is False or there are no secondary labels, returns a one-hot vector for primary and any secondaries (multi-hot if multiple and not soft).
    """
    num_classes = len(class_map)
    label_vec = np.zeros(num_classes, dtype=np.float32)
    if not secondary_labels or not use_soft:
        # No secondaries or we want hard labels
        # Mark primary as 1
        if primary_label in class_map:
            label_vec[class_map[primary_label]] = 1.0
        # If there are secondary labels but no soft labeling, mark them as well (multi-hot)
        if secondary_labels and not use_soft:
            for sec in secondary_labels:
                if sec in class_map:
                    label_vec[class_map[sec]] = 1.0
    else:
        # Soft label distribution: primary and secondaries
        if primary_label in class_map:
            label_vec[class_map[primary_label]] = primary_weight
        if secondary_labels:
            # distribute secondary_weight among all secondary labels
            share = secondary_weight
            sec_count = len(secondary_labels)
            if sec_count > 0:
                per_sec = share / sec_count
            else:
                per_sec = 0.0
            for sec in secondary_labels:
                if sec in class_map:
                    label_vec[class_map[sec]] = per_sec
    return label_vec

def hash_chunk_id(filename: str, start_sec: float) -> str:
    """
    Generate a unique hash ID for a chunk given the source filename and start time.
    """
    base = f"{filename}_{start_sec:.3f}"
    # Use a short hash for uniqueness
    h = hashlib.sha1(base.encode('utf-8')).hexdigest()
    return h[:8]  # use first 8 hex digits for brevity

def resize_mel(mel_db: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize a mel spectrogram (in dB) to the target height and width using bilinear interpolation.
    Preserves dynamic range by normalizing before resize and re-scaling after.
    """
    h, w = mel_db.shape
    if (h, w) == (target_h, target_w):
        return mel_db
    mel_min, mel_max = mel_db.min(), mel_db.max()
    # Normalize to 0-255
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
    mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))
    mel_img = mel_img.resize((target_w, target_h), Image.BILINEAR)
    mel_resized = np.array(mel_img).astype(np.float32) / 255.0
    # Re-scale to original dB range
    mel_resized = mel_resized * (mel_max - mel_min + 1e-6) + mel_min
    return mel_resized

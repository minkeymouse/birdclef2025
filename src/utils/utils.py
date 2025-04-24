#!/usr/bin/env python3
"""
utils.py – shared helpers for the BirdCLEF-2025 pipeline
========================================================
Functions here are imported by *every* stage (preprocessing, training, inference).
Keep them lightweight: **no heavy ML imports at module load**.

Provided helpers
----------------
load_taxonomy         → canonical class list + {species: idx} map (cached)
parse_secondary_labels → robust '[…]'-string → List[str] parser
create_label_vector   → one-hot / soft-label vector, handles strings or lists
hash_chunk_id         → short SHA-1 from (filename, start_sec)
resize_mel            → bilinear resize that preserves dB range
load_vad              → lazy Silero VAD loader (torch-hub)

Public API is defined in __all__ at bottom.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch, torchaudio

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Taxonomy utilities
# ----------------------------------------------------------------------------
_PRI_LABEL_COLUMNS = ("primary_label", "ebird_code", "species_code")

@lru_cache(maxsize=1)
def load_taxonomy(
    taxonomy_csv: Optional[Union[str, Path]],
    train_csv: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Load species codes from taxonomy CSV or fallback to train CSV.
    Returns (sorted class_list, class_map).
    """
    def _extract(df: pd.DataFrame) -> Sequence[str]:
        for col in _PRI_LABEL_COLUMNS:
            if col in df.columns:
                return df[col].dropna().astype(str).unique()
        raise ValueError(f"Expected one of {_PRI_LABEL_COLUMNS} in columns.")

    if taxonomy_csv and Path(taxonomy_csv).is_file():
        df = pd.read_csv(str(taxonomy_csv))
        species = _extract(df)
    elif train_csv and Path(train_csv).is_file():
        df = pd.read_csv(str(train_csv))
        species = _extract(df)
    else:
        raise FileNotFoundError(
            "Could not read taxonomy or train CSV for species list."
        )

    class_list = sorted(map(str, species))
    class_map = {sp: idx for idx, sp in enumerate(class_list)}
    return class_list, class_map

# ----------------------------------------------------------------------------
# Label helpers
# ----------------------------------------------------------------------------
def parse_secondary_labels(
    sec: Optional[Union[str, Sequence[str]]]
) -> List[str]:
    """
    Parse BirdCLEF 'secondary_labels' entries into List[str].
    Returns empty list if none or unparsable.
    """
    if sec is None:
        return []
    if isinstance(sec, float) and np.isnan(sec):
        return []
    if isinstance(sec, list):
        return [str(s).strip() for s in sec if s]
    if isinstance(sec, str):
        s = sec.strip()
        if not s or s == "[]":
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if x]
            if isinstance(parsed, str):
                return [parsed.strip()]
        except Exception:
            return [x.strip() for x in s.replace(",", " ").split() if x.strip()]
    return []


def create_label_vector(
    primary_label: str,
    secondary_labels: Optional[Union[str, Sequence[str]]],
    class_map: Dict[str, int],
    *,
    primary_weight: float = 1.0,
    secondary_weight: float = 0.05,
    use_soft: bool = True,
) -> np.ndarray:
    """
    Build a target vector (len = len(class_map)) for training:
    primary label = primary_weight, secondaries = equal share of secondary_weight.
    If use_soft=False, secondaries get 1.0 (multi-hot).
    """
    vec = np.zeros(len(class_map), dtype=np.float32)
    # Primary
    if primary_label in class_map:
        vec[class_map[primary_label]] = primary_weight
    # Secondary
    secs = parse_secondary_labels(secondary_labels)
    if not secs:
        return vec
    if use_soft:
        share = secondary_weight / len(secs)
        for sp in secs:
            if sp in class_map:
                vec[class_map[sp]] = share
    else:
        for sp in secs:
            if sp in class_map:
                vec[class_map[sp]] = 1.0
    return vec

# ----------------------------------------------------------------------------
# Misc small helpers
# ----------------------------------------------------------------------------
def hash_chunk_id(
    filename: str,
    start_sec: float,
    length: int = 8,
) -> str:
    """
    Short SHA-1 hash for (filename, start_sec).
    """
    txt = f"{filename}_{start_sec:.3f}".encode()
    return hashlib.sha1(txt).hexdigest()[:length]


def resize_mel(
    mel_db: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Resize log-mel spectrogram preserving dynamic range.
    """
    h, w = mel_db.shape
    if (h, w) == (target_h, target_w):
        return mel_db
    lo, hi = float(mel_db.min()), float(mel_db.max())
    norm = (mel_db - lo) / (hi - lo + 1e-6)
    img = Image.fromarray((norm * 255).astype(np.uint8))
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr * (hi - lo) + lo

# ----------------------------------------------------------------------------
# Voice-activity detection (lazy torch-hub)
# ----------------------------------------------------------------------------

def is_silent(
    wave: np.ndarray,
    thresh_db: float = -50.0,
) -> bool:
    """
    Check if the audio is silent based on a dB threshold.
    """
    db = 10 * np.log10(np.maximum(1e-12, np.mean(wave**2)))
    return db < thresh_db

def contains_voice(
    samples: np.ndarray,
    vad_model: torch.jit.ScriptModule,
    get_speech_timestamps: Callable,
) -> bool:
    """
    Check if the audio contains voice using VAD.
    """
    ts = get_speech_timestamps(samples, vad_model, sampling_rate=16000, threshold=0.5)
    return bool(ts)

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
__all__ = [
    "load_taxonomy",
    "parse_secondary_labels",
    "create_label_vector",
    "hash_chunk_id",
    "resize_mel",
    "load_vad",
]

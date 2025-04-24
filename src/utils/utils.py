#!/usr/bin/env python3
"""
utils.py – shared helpers for the BirdCLEF-2025 pipeline
========================================================
Functions here are imported by *every* stage (pre-processing, training,
inference).  Keep them lightweight: **no heavy ML imports at module load**.

Provided helpers
----------------
load_taxonomy         → canonical class list + {species: idx} map (cached)
parse_secondary_labels → robust '[…]'-string → List[str] parser
create_label_vector   → one-hot / soft-label vector, handles strings or lists
hash_chunk_id         → short SHA-1 from (filename, start_sec)
resize_mel            → bilinear resize that preserves dB range
load_vad              → lazy Silero VAD loader (torch-hub)

All public names are re-exported via ``utils.__init__.__all__``.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------------------------------------------------------#
# Taxonomy utilities                                                           #
# -----------------------------------------------------------------------------#
_PRI_LABEL_COLUMNS = ("primary_label", "ebird_code", "species_code")


@lru_cache(maxsize=1)
def load_taxonomy(
    taxonomy_csv_path: str | os.PathLike | None,
    train_csv_path: str | os.PathLike | None = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Return *(class_list, class_map)* where ``class_list`` is the **sorted**
    list of all 206 species codes and ``class_map`` maps code → index.

    The lookup order is

        1. explicit *taxonomy.csv* (preferred)
        2. fallback to *train.csv* primary labels.
    """
    def _read_species_from_df(df: pd.DataFrame) -> Sequence[str]:
        for col in _PRI_LABEL_COLUMNS:
            if col in df.columns:
                return df[col].dropna().astype(str).unique()
        raise ValueError(
            "None of the expected label columns "
            f"{_PRI_LABEL_COLUMNS} found in DataFrame."
        )

    # ---------- 1) taxonomy CSV ------------------------------------------------
    if taxonomy_csv_path and Path(taxonomy_csv_path).is_file():
        species = _read_species_from_df(pd.read_csv(taxonomy_csv_path))
    # ---------- 2) fallback: train CSV ----------------------------------------
    elif train_csv_path and Path(train_csv_path).is_file():
        species = _read_species_from_df(pd.read_csv(train_csv_path))
    else:
        raise FileNotFoundError(
            "Neither taxonomy_csv_path nor train_csv_path could be read."
        )

    class_list = sorted(map(str, species))
    class_map = {sp: i for i, sp in enumerate(class_list)}
    return class_list, class_map


# -----------------------------------------------------------------------------#
# Label helpers                                                                #
# -----------------------------------------------------------------------------#
def parse_secondary_labels(sec: str | Sequence[str] | None) -> List[str]:
    """
    Convert BirdCLEF’s *secondary_labels* column (often a stringified list) to
    ``List[str]``.  Returns ``[]`` if empty / None / unparsable.
    """
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return []
    if isinstance(sec, list):
        return [str(s).strip() for s in sec if s]
    if isinstance(sec, str):
        sec = sec.strip()
        if sec in ("", "[]"):
            return []
        try:
            parsed = ast.literal_eval(sec)
            # literal_eval can return tuple / list / str
            if isinstance(parsed, (list, tuple)):
                return [str(s).strip() for s in parsed if s]
            if isinstance(parsed, str):
                return [parsed.strip()]
        except Exception:
            # fallback: assume comma / space separated
            return [s.strip() for s in sec.replace(",", " ").split() if s.strip()]
    return []


def create_label_vector(
    primary_label: str,
    secondary_labels: Sequence[str] | str | None,
    class_map: Dict[str, int],
    *,
    primary_weight: float = 0.7,
    secondary_weight: float = 0.3,
    use_soft: bool = True,
) -> np.ndarray:
    """
    Build a label vector of length ``len(class_map)``.

    Parameters
    ----------
    primary_label
        The *primary_label* field from train.csv.
    secondary_labels
        Sequence or raw string from *secondary_labels* column.
    class_map
        Species → index mapping (from :func:`load_taxonomy`)
    primary_weight, secondary_weight
        Weights used when ``use_soft=True`` *and* secondaries exist.
        They **do not have to sum to 1.0** – this vector is treated as
        a target *probability* distribution, not necessarily normalised.
    use_soft
        If *False*, any present secondary label(s) get **1.0** just like
        the primary (multi-hot).  If *True*, uses the provided weights.

    Notes
    -----
    *Any* unknown species codes are silently ignored (rare but safe).
    """
    sec_list = parse_secondary_labels(secondary_labels)
    n_classes = len(class_map)
    vec = np.zeros(n_classes, dtype=np.float32)

    # -------- primary ---------------------------------------------------------
    if primary_label in class_map:
        vec[class_map[primary_label]] = (
            primary_weight if (sec_list and use_soft) else 1.0
        )

    # -------- secondary -------------------------------------------------------
    if not sec_list:
        return vec

    if use_soft:
        per_sec = secondary_weight / len(sec_list)
        for sp in sec_list:
            if sp in class_map:
                vec[class_map[sp]] = per_sec
    else:  # multi-hot
        for sp in sec_list:
            if sp in class_map:
                vec[class_map[sp]] = 1.0
    return vec


# -----------------------------------------------------------------------------#
# Misc small helpers                                                           #
# -----------------------------------------------------------------------------#
def hash_chunk_id(filename: str, start_sec: float, length: int = 8) -> str:
    """
    Short SHA-1 hash for a (file, start_time) pair – good enough for 10⁶+ chunks.

    >>> hash_chunk_id("XC123.ogg", 12.345)
    'e1a2b3c4'
    """
    h = hashlib.sha1(f"{filename}_{start_sec:.3f}".encode()).hexdigest()
    return h[:length]


def resize_mel(
    mel_db: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Bilinear-resize a log-mel spectrogram **without destroying its dynamic
    range** (normalise → resize → de-normalise).
    """
    h, w = mel_db.shape
    if (h, w) == (target_h, target_w):
        return mel_db

    lo, hi = float(mel_db.min()), float(mel_db.max())
    norm = (mel_db - lo) / (hi - lo + 1e-6)
    img = Image.fromarray((norm * 255).astype(np.uint8))
    img = img.resize((target_w, target_h), Image.BILINEAR)
    out = np.asarray(img).astype(np.float32) / 255.0
    return out * (hi - lo) + lo


# -----------------------------------------------------------------------------#
# Voice-activity detection (lazy torch-hub)                                    #
# -----------------------------------------------------------------------------#
def load_vad(cache_dir: str | os.PathLike | None = None):
    """
    Lazy-load the **Silero VAD** Torch-Script model.

    Returns
    -------
    model : torch.jit.ScriptModule
    helpers : dict[str, callable]
        Keys: ``get_speech_timestamps``, ``read_audio``,
        ``save_audio``, ``collect_chunks``, ``merge_chunks``
    """
    import importlib
    import types

    torch = importlib.import_module("torch")  # heavy, but only *inside* the call
    if cache_dir:
        os.environ.setdefault("TORCH_HOME", str(Path(cache_dir).expanduser()))

    logging.getLogger("torch.hub").setLevel(logging.ERROR)
    model, utils_tuple = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        trust_repo=True,
        onnx=False,
        force_reload=False,
    )
    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = utils_tuple

    helpers = {
        "get_speech_timestamps": get_speech_timestamps,
        "save_audio": save_audio,
        "read_audio": read_audio,
        "collect_chunks": collect_chunks,
        # merge_chunks was removed upstream; VADIterator covers it
        "VADIterator": VADIterator,
    }
    return model, helpers


# -----------------------------------------------------------------------------#
# Public re-exports (duplicated in utils/__init__.py)                          #
# -----------------------------------------------------------------------------#
__all__ = [
    "load_taxonomy",
    "parse_secondary_labels",
    "create_label_vector",
    "hash_chunk_id",
    "resize_mel",
    "load_vad",
]

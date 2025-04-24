"""utils package  – shared helpers for the BirdCLEF 2025 pipeline
================================================================
This package gathers the lightweight, re‑usable helpers that every
processing stage imports.  Everything is re‑exported here so callers can
simply write::

    from utils import load_taxonomy, hash_chunk_id, resize_mel, load_vad

Design notes
------------
* **Lazy heavy imports** – `load_vad()` handles the Torch hub call on first
  invocation, so importing `utils` never pulls in `torch` unless it’s needed.
* **Single public surface** – by re‑exporting the helpers via `__all__`, the
  location of the actual implementation (`utils.py`, `audio.py`, …) can change
  later without breaking import sites.
"""
from __future__ import annotations

# Re‑export public helpers from the implementation module ---------------------
from .utils import (
    load_taxonomy,
    create_label_vector,
    hash_chunk_id,
    resize_mel,
    load_vad,
)

__all__ = [
    "load_taxonomy",
    "create_label_vector",
    "hash_chunk_id",
    "resize_mel",
    "load_vad",
]

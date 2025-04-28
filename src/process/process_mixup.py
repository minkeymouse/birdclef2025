#!/usr/bin/env python3
"""
process_mixup.py ― Generate **multi‑species mix‑up** chunks for BirdCLEF‑2025
==============================================================================

**New feature** (2025‑04‑26): each mixed chunk now overlays a *random* number
of recordings **uniformly sampled from 2 to 5**.  This better mimics real
soundscapes where several species may vocalise simultaneously.

Pipeline (updated)
------------------
1. Balance the candidate pool so no primary species exceeds 20 % share and all
   206 species appear at least once.
2. Coarse geo‑cluster recordings on a 0.5° grid.
3. Within every cluster, repeatedly draw **k ∼ U{2, 3, 4, 5}** distinct
   recordings with different primary species until <2 remain.
4. Overlay the *k* waves at equal power → `mix = (1/k)·Σ yᵢ`, peak‑normalise.
5. Skip silent or speech‑contaminated chunks; slide a 10 s window (5 s hop),
   save log‑mel and multi‑label vector (primary = 1.0; others = 0.8·rating/5).
6. Append rows to `DATABASE/train_metadata.csv`; update `audio_hashes.txt`.

Run
---
```bash
python -m src.process.process_mixup          # writes files
python -m src.process.process_mixup --dry-run # logic only, no writes
```
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils import utils  # noqa: E402

with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
audio_cfg = CFG["audio"]
chunk_cfg = CFG["chunking"]
mel_cfg = CFG["mel"]
sel_cfg = CFG["selection"]
label_cfg = CFG["labeling"]

audio_dir = Path(paths_cfg["audio_dir"])
processed_dir = Path(paths_cfg["processed_dir"])
mel_dir = processed_dir / "mels"
label_dir = processed_dir / "labels"
for d in (mel_dir, label_dir):
    d.mkdir(parents=True, exist_ok=True)

metadata_csv = Path(paths_cfg["train_metadata"])
train_csv = Path(paths_cfg["train_csv"])
hash_file = processed_dir / "audio_hashes.txt"

parser = argparse.ArgumentParser(description="Generate mix‑up training chunks")
parser.add_argument("--dry-run", action="store_true", help="Skip disk writes")
args = parser.parse_args()
DRY = args.dry_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_mixup")

CLASS_LIST, CLASS_MAP = utils.load_taxonomy(paths_cfg.get("taxonomy_csv"), train_csv)
NUM_CLASSES = len(CLASS_LIST)
SEEN_HASHES: set[str] = set(hash_file.read_text().splitlines()) if hash_file.exists() else set()

SR = audio_cfg["sample_rate"]
CHUNK_S = chunk_cfg["train_chunk_duration"]
HOP_S = chunk_cfg["train_chunk_hop"]
CHUNK_SAMPLES = int(CHUNK_S * SR)
HOP_SAMPLES = int(HOP_S * SR)
MIN_DUR_SAMPLES = int(audio_cfg["min_duration"] * SR)
SILENCE_DB = audio_cfg.get("silence_thresh_db", -50.0)
TRIM_DB = audio_cfg.get("trim_top_db")

MIX_SEC_WEIGHT = 0.8  # weight for non‑primary species

# -----------------------------------------------------------------------------
# Stage 1 – Build balanced candidate pool
# -----------------------------------------------------------------------------
df = pd.read_csv(train_csv)
min_rating = sel_cfg.get("minimum_rating", 5)
rare_thresh = sel_cfg.get("rare_species_threshold", 100)
max_count = sel_cfg.get("max_count", 300)

primary_counts = df.groupby("primary_label")["filename"].transform("count")
minority_df = df[primary_counts < rare_thresh]
good_df = df[df["rating"] >= min_rating]
not_too_much_df = df[primary_counts < max_count]
base_df = pd.concat([minority_df, good_df, not_too_much_df], ignore_index=True)

cap_pct = 0.20
label_caps = {
    sp: math.ceil(cap_pct * len(base_df)) if cnt / len(base_df) > cap_pct else cnt
    for sp, cnt in base_df["primary_label"].value_counts().items()
}
subsets = [
    base_df[base_df["primary_label"] == sp].sample(n=cap, random_state=None, replace=False)
    for sp, cap in label_caps.items()
]
pool_df = pd.concat(subsets, ignore_index=True).drop_duplicates("filename")
missing = set(CLASS_LIST) - set(pool_df["primary_label"].unique())
if missing:
    log.warning("Added %d missing species for full coverage", len(missing))
    add_df = (
        df[df["primary_label"].isin(missing)]
        .sort_values("rating", ascending=False)
        .groupby("primary_label")
        .head(1)
    )
    pool_df = pd.concat([pool_df, add_df], ignore_index=True)

# -----------------------------------------------------------------------------
# Stage 2 – geo‑clusters (0.5° grid)
# -----------------------------------------------------------------------------

def cluster_id(lat: float, lon: float, gran: float = 0.5) -> str:
    if np.isnan(lat) or np.isnan(lon):
        return "nan"
    return f"{round(lat / gran) * gran:.1f}_{round(lon / gran) * gran:.1f}"

pool_df["cluster"] = pool_df.apply(lambda r: cluster_id(r.get("latitude", np.nan), r.get("longitude", np.nan)), axis=1)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_clip(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if TRIM_DB is not None:
        y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    if y.size < MIN_DUR_SAMPLES:
        reps = math.ceil(MIN_DUR_SAMPLES / y.size)
        y = np.tile(y, reps)[:MIN_DUR_SAMPLES]
    return y.astype(np.float32)


def overlay(waves: List[np.ndarray]) -> np.ndarray:
    max_len = max(w.size for w in waves)
    stacked = np.zeros(max_len, dtype=np.float32)
    for w in waves:
        if w.size < max_len:
            w = np.pad(w, (0, max_len - w.size))
        stacked += w / len(waves)
    return stacked / (np.max(np.abs(stacked)) + 1e-6)

# -----------------------------------------------------------------------------
# Stage 3 – assemble multi‑clip groups & generate chunks
# -----------------------------------------------------------------------------
meta_rows: List[dict] = []
new_hashes: set[str] = set()
clusters = {k: g.sample(frac=1.0, random_state=42) for k, g in pool_df.groupby("cluster") if k != "nan"}

MAX_NEW = 3000
generated = 0

for g in clusters.values():
    remain = g.to_dict("records")
    random.shuffle(remain)
    while len(remain) >= 2:
        k = random.randint(2, min(5, len(remain)))  # 2 – 5 clips
        group = [remain.pop() for _ in range(k)]
        waves, labels, ratings = [], [], []
        duplicate_species = False
        seen_sp: set[str] = set()
        for rec in group:
            sp = rec["primary_label"]
            if sp in seen_sp:
                duplicate_species = True
                break
            seen_sp.add(sp)
            path = audio_dir / rec["filename"]
            if not path.is_file():
                duplicate_species = True
                break
            waves.append(load_clip(path))
            labels.append(sp)
            ratings.append(rec["rating"])
        if duplicate_species:
            continue
        mix = overlay(waves)
        if utils.is_silent(mix, db_thresh=SILENCE_DB) or utils.contains_voice(mix, SR):
            continue

        ptr = 0
        while ptr + CHUNK_SAMPLES <= mix.size:
            chunk = mix[ptr : ptr + CHUNK_SAMPLES]
            ptr += HOP_SAMPLES
            mel = librosa.feature.melspectrogram(
                y=chunk,
                sr=SR,
                n_fft=mel_cfg["n_fft"],
                hop_length=mel_cfg["hop_length"],
                n_mels=mel_cfg["n_mels"],
                fmin=mel_cfg["fmin"],
                fmax=mel_cfg["fmax"],
                power=mel_cfg["power"],
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = utils.resize_mel(mel_db, *mel_cfg["target_shape"]).astype(np.float32)

            cid_str = "|".join([rec["filename"] for rec in group]) + f"_{ptr/SR:.3f}"
            cid = hashlib.sha1(cid_str.encode()).hexdigest()[:8]
            if cid in SEEN_HASHES or cid in new_hashes:
                continue
            new_hashes.add(cid)

            mel_path = mel_dir / f"{cid}.npy"
            label_path = label_dir / f"{cid}.npy"
            if not DRY:
                np.save(mel_path, mel_db)

            lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
            # weight scheme: first = 1.0, others = 0.8·rating/5
            for i, sp in enumerate(labels):
                idx = CLASS_MAP.get(sp)
                if idx is None:
                    continue
                w = 1.0 if i == 0 else MIX_SEC_WEIGHT * (ratings[i] / 5.0)
                lbl[idx] = max(lbl[idx], w)
            if not DRY:
                np.save(label_path, lbl)
            generated += 1
            if generated >= MAX_NEW:
                break

            meta_rows.append(
                {
                    "filename": "|".join([rec["filename"] for rec in group]),
                    "end_sec": round(ptr / SR, 3),
                    "mel_path": str(mel_path),
                    "label_path": str(label_path),
                    "weight": float(label_cfg.get("rare_label_weight", 1.0)),
                    "source": "mixup_audio",
                    "species": "|".join(labels),
                }
            )

log.info("Generated %d mixed chunks", len(meta_rows))

if meta_rows and not DRY:
    meta_df = pd.DataFrame(meta_rows)
    if metadata_csv.exists():
        old = pd.read_csv(metadata_csv)
        meta_df = pd.concat([old, meta_df], ignore_index=True)
    meta_df.to_csv(metadata_csv, index=False)
    SEEN_HASHES.update(new_hashes)
    with hash_file.open("w") as f:
        f.write("\n".join(sorted(SEEN_HASHES)))
    log.info("Appended metadata and updated hash list")
else:
    log.info("Dry‑run complete – no files written.")

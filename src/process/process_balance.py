#!/usr/bin/env python3
"""
process_balance.py ― down-sample over-represented *train_audio* species
=======================================================================

Keeps mix-ups, pseudo-labels, soundscapes, etc. intact; only the original
chunks whose `source` is NA or "train_audio" are pruned so that no single
species exceeds a given proportion (default 5 %).
"""
from __future__ import annotations
import argparse, logging, sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import yaml

# ───────────────────────────── CLI ─────────────────────────────
p = argparse.ArgumentParser(description="Limit each species to ≤ cap% of rows")
p.add_argument("--cap",  type=float, default=0.02,
               help="maximum share per species (0-1, default 0.05)")
p.add_argument("--dry-run", action="store_true",
               help="preview only – don’t delete files / rewrite CSV")
args  = p.parse_args()
CAP   = max(1e-6, min(args.cap, 1.0))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_balance")

# ─────────────────── paths & config ───────────────────
root = Path(__file__).resolve().parents[2]
with open(root / "config" / "process.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

meta_csv  = root / CFG["paths"]["train_metadata"]
train_csv = root / CFG["paths"]["train_csv"]

# ───────────────── load metadata ────────────────────────
meta      = pd.read_csv(meta_csv)                # LABEL METADATA
train_df  = pd.read_csv(train_csv,               # CANONICAL LABELS
                        usecols=["filename", "primary_label"])

# ← UPDATED — ensure **exactly one** primary_label column with no NaNs
if "primary_label" not in meta.columns:
    meta = meta.merge(train_df, on="filename", how="left",
                      validate="many_to_one")
else:
    miss = meta["primary_label"].isna()
    if miss.any():                               # back-fill the gaps
        meta = meta.merge(train_df, on="filename", how="left",
                          suffixes=("", "_train"), validate="many_to_one")
        meta["primary_label"] = meta["primary_label"].fillna(
            meta["primary_label_train"])
        meta.drop(columns=["primary_label_train"], inplace=True)

if meta["primary_label"].isna().all():
    log.error("No primary labels could be assigned – aborting.")
    sys.exit(1)
# ─────────────────── balancing logic ───────────────────
ta_mask = meta["source"].isna() | (meta["source"] == "train_audio")
ta_df   = meta[ta_mask].copy()
other   = meta[~ta_mask]

if ta_df.empty:
    log.info("No train_audio rows found – nothing to balance.")
    sys.exit(0)

total   = len(ta_df)
cap_n   = int(np.ceil(CAP * total))
rows_to_drop: List[int] = []

for sp, cnt in ta_df["primary_label"].value_counts().items():
    if cnt > cap_n:
        excess = cnt - cap_n
        rows_to_drop.extend(
            ta_df[ta_df["primary_label"] == sp]      # rows of that species
                 .sample(n=excess, random_state=42).index)

log.info("Will remove %d/%d train_audio rows (%.1f %%)",
         len(rows_to_drop), total, 100 * len(rows_to_drop) / total)

if args.dry_run:
    log.info("[dry-run] No files deleted, CSV unchanged.")
    sys.exit(0)

# ───────── delete corresponding mel/label files ─────────
deleted: Dict[str, int] = defaultdict(int)
for idx in rows_to_drop:
    row = meta.loc[idx]
    for col in ("mel_path", "label_path"):
        try:
            Path(row[col]).unlink()
            deleted[col] += 1
        except FileNotFoundError:
            pass

# ─────────── write balanced metadata CSV ────────────────
balanced = pd.concat([ta_df.drop(index=rows_to_drop), other],
                     ignore_index=True)
balanced.to_csv(meta_csv, index=False)

log.info("Deleted %d mel and %d label files; new metadata has %d rows.",
         deleted["mel_path"], deleted["label_path"], len(balanced))

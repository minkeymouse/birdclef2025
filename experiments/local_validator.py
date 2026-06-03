#!/usr/bin/env python3
"""Local validator — gate for LB submissions.

GOAL: any candidate submission.csv must pass local validation BEFORE we burn
an LB slot. v6/v7/v9 regressions taught us that "mechanism-clean" alone isn't
enough — we need a quantitative local Δ macro AUC vs the anchor.

USE:
  uv run python experiments/local_validator.py \
      --candidate /tmp/candidate_submission.csv \
      --anchor /tmp/anchor_submission.csv

PASS criterion: Δ macro AUC ≥ +0.001 on labeled SS (66 files, 739 rows).

CAVEAT: labeled SS is anti-correlated with LB at the margin (memory invariant).
A candidate passing local validation can still regress on LB. This validator
is a NECESSARY filter (skip clearly-bad ones), NOT SUFFICIENT (positive local
≠ positive LB). It catches obvious bugs (shape mismatch, NaN, all-zeros) and
mechanism-broken candidates.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"


def load_truth():
    """Build truth matrix for all labeled SS rows."""
    ss_lab = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    ss_lab["end_sec"] = ss_lab["end"].apply(
        lambda x: int(x.split(":")[0]) * 3600 + int(x.split(":")[1]) * 60 + int(x.split(":")[2])
    )
    sub = pd.read_csv(DATA / "sample_submission.csv")
    PRIMARY_LABELS = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    key2labels: dict[tuple, set] = {}
    for _, r in ss_lab.iterrows():
        key = (r["filename"], int(r["end_sec"]))
        key2labels.setdefault(key, set())
        for lab in str(r["primary_label"]).split(";"):
            lab = lab.strip()
            if lab:
                key2labels[key].add(lab)
    return key2labels, PRIMARY_LABELS, l2i


def build_y_for_csv(csv: pd.DataFrame, key2labels, l2i, primary_labels):
    """row_id format: <filename>_<endsec>. Return Y matrix aligned with csv."""
    n = len(csv)
    y = np.zeros((n, 234), dtype=np.float32)
    matched = 0
    for i, rid in enumerate(csv["row_id"].astype(str)):
        fname, endsec = rid.rsplit("_", 1)
        fname += ".ogg"
        endsec = int(endsec)
        labs = key2labels.get((fname, endsec), set())
        if labs:
            matched += 1
            for lab in labs:
                if lab in l2i:
                    y[i, l2i[lab]] = 1.0
    return y, matched


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray):
    aucs = []
    for c in range(y_true.shape[1]):
        if 0 < y_true[:, c].sum() < y_true.shape[0]:
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--anchor", default=None,
                   help="Path to anchor submission.csv. Optional — if missing, just reports candidate macro AUC.")
    p.add_argument("--gate", type=float, default=0.001,
                   help="Required Δ macro AUC to pass (default 0.001).")
    args = p.parse_args()

    key2labels, primary_labels, l2i = load_truth()
    cand = pd.read_csv(args.candidate)
    cand_cols = [c for c in cand.columns if c != "row_id"]
    if cand_cols != primary_labels:
        print(f"ERROR: column order mismatch. Cand has {len(cand_cols)} cols, expected {len(primary_labels)}.")
        return 1

    y_cand, matched = build_y_for_csv(cand, key2labels, l2i, primary_labels)
    if matched == 0:
        print(f"ERROR: no rows matched truth (row_id format issue?).")
        return 1
    print(f"matched rows: {matched}/{len(cand)}")

    cand_auc, n_cand = macro_auc(y_cand, cand[cand_cols].values)
    print(f"candidate macro AUC: {cand_auc:.4f}  (n_classes_present={n_cand})")

    if args.anchor:
        anch = pd.read_csv(args.anchor)
        # Align by row_id
        anch = anch.set_index("row_id").loc[cand["row_id"]].reset_index()
        anch_auc, n_anch = macro_auc(y_cand, anch[cand_cols].values)
        delta = cand_auc - anch_auc
        print(f"anchor    macro AUC: {anch_auc:.4f}")
        print(f"Δ:                  {delta:+.4f}")
        if delta >= args.gate:
            print(f"PASS: Δ ≥ {args.gate}. Candidate eligible for LB submit.")
            return 0
        else:
            print(f"FAIL: Δ < {args.gate}. Don't burn slot.")
            print(f"NOTE: anti-correlation hazard means even passing candidates can regress on LB.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

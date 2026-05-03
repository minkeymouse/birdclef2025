#!/usr/bin/env python3
"""exp169 investigation: per-taxon AUC of fold 0 ckpt vs Tucker 5-fold and exp50.

Goal: see whether our distilled SED matches Tucker on non-Aves taxa
(Insecta/Mammalia/Amphibia) — the place where Perch's bird-only pretraining
might leave a gap. Also confirm same-site (labeled SS eval) calibration.

Loads fold 0 ckpt (val_SS = 0.8411 reported) and runs per-taxon AUC on
the eval portion of labeled SS using the cached waveforms (CPU-only).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"

sys.path.insert(0, str(ROOT / "experiments" / "_data_pipelines"))
from exp169_distilled_sed import DistilledSED, build_primaries, EVAL_SS_N_FILES, SEED  # noqa


def load_fold(fold: int, dev: str = "cpu") -> DistilledSED:
    ck = CACHE_DIR / f"fold{fold}" / "best_ckpt.pt"
    state = torch.load(ck, map_location=dev)
    m = DistilledSED(n_cls=234)
    m.load_state_dict(state["state_dict"])
    m.eval().to(dev)
    return m, state


@torch.no_grad()
def predict_indices(model, indices, wav_mm, dev, batch=16):
    n_cls = 234
    out = np.zeros((len(indices), n_cls), dtype=np.float32)
    for i in range(0, len(indices), batch):
        sl = indices[i:i + batch]
        wavs = wav_mm[sl].astype(np.float32)
        x = torch.from_numpy(wavs).to(dev)
        clip_l, _, _ = model(x)
        out[i:i + len(sl)] = torch.sigmoid(clip_l).cpu().numpy()
    return out


def main():
    primary, l2i = build_primaries()
    n_cls = len(l2i)
    tax = pd.read_csv(DATA / "taxonomy.csv")
    cls_to_taxon = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    taxon_idx = {t: [l2i[c] for c in cls_to_taxon if cls_to_taxon[c] == t and c in l2i]
                 for t in tax["class_name"].unique()}
    print("taxon class counts:", {t: len(v) for t, v in taxon_idx.items()})

    print("loading cache (memmap)...")
    wav_mm = np.load(CACHE_DIR / "waveforms_fp16.npy", mmap_mode="r")
    meta = np.load(CACHE_DIR / "meta.npz", allow_pickle=True)
    filenames = meta["filenames"]
    is_ss = meta["is_ss"]
    ss_label_str = meta["ss_label_str"]

    ss_files = sorted({filenames[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(SEED)
    rng.shuffle(ss_files)
    eval_files = set(ss_files[:EVAL_SS_N_FILES])
    eval_indices = np.array(
        [i for i in np.where(is_ss == 1)[0] if filenames[i] in eval_files],
        dtype=np.int64,
    )
    Y = np.zeros((len(eval_indices), n_cls), dtype=np.uint8)
    for i, gi in enumerate(eval_indices):
        s = ss_label_str[gi]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        for lbl in str(s).split(";"):
            lbl = lbl.strip()
            if lbl in l2i:
                Y[i, l2i[lbl]] = 1
    print(f"labeled SS eval rows: {len(eval_indices)}")
    print(f"  positives per taxon: ", end="")
    for t, idxs in taxon_idx.items():
        if not idxs: continue
        n_pos = (Y[:, idxs].sum(axis=0) > 0).sum()
        n_total = len(idxs)
        print(f"{t}={n_pos}/{n_total}", end=" ")
    print()

    print("loading fold 0 ckpt...")
    t0 = time.time()
    m0, _ = load_fold(0, dev="cpu")
    preds = predict_indices(m0, eval_indices, wav_mm, dev="cpu")
    print(f"predict done in {time.time()-t0:.0f}s  shape {preds.shape}")

    # per-class AUC
    aucs = np.full(n_cls, np.nan)
    for c in range(n_cls):
        s = Y[:, c].sum()
        if s == 0 or s == len(Y):
            continue
        try:
            aucs[c] = roc_auc_score(Y[:, c], preds[:, c])
        except Exception:
            pass

    print("\n=== Per-taxon AUC (exp169 fold 0) on labeled SS eval ===")
    rows = []
    for t, idxs in taxon_idx.items():
        if not idxs: continue
        ts = np.array(idxs)
        valid = ~np.isnan(aucs[ts])
        n_eval = valid.sum()
        if n_eval == 0:
            rows.append((t, len(idxs), 0, np.nan, np.nan))
            continue
        ts_v = ts[valid]
        rows.append((t, len(idxs), int(n_eval), float(np.nanmean(aucs[ts_v])), float(np.nanmedian(aucs[ts_v]))))
    rows.sort(key=lambda r: -r[1])
    print(f"{'taxon':12s} {'n_cls':>6s} {'n_eval':>7s} {'mean_auc':>9s} {'median_auc':>11s}")
    for t, nc, ne, ma, mm in rows:
        print(f"{t:12s} {nc:>6d} {ne:>7d} {ma:>9.4f} {mm:>11.4f}")

    macro = float(np.nanmean(aucs))
    print(f"\nmacro AUC (skip empty): {macro:.4f}")

    # save
    out = {
        "fold": 0,
        "macro_auc_skip_empty": macro,
        "per_taxon": [
            {"taxon": t, "n_cls": nc, "n_eval": ne,
             "mean_auc": (ma if np.isfinite(ma) else None),
             "median_auc": (mm if np.isfinite(mm) else None)}
            for t, nc, ne, ma, mm in rows
        ],
    }
    out_dir = ROOT / "experiments" / "_audits_post_v26" / "exp169_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fold0_taxon_auc.json").write_text(json.dumps(out, indent=2))
    print(f"saved {out_dir / 'fold0_taxon_auc.json'}")


if __name__ == "__main__":
    main()

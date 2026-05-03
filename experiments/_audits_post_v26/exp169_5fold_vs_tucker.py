#!/usr/bin/env python3
"""5-fold exp169 vs 5-fold Tucker — fair comparison and ensemble simulation.

The fold-0 vs Tucker comparison was unfair because Tucker trained on the
labeled-SS rows we held out for our own training. To get a fair signal, we
predict with the exp169 5-fold ENSEMBLE on the same eval rows and:

  1. Compare per-taxon AUC (still biased toward Tucker on shared rows).
  2. Compute the per-class Pearson correlation between the two ensembles
     (this is unbiased — it measures structure agreement regardless of
     either being train-set fitted).
  3. Simulate three blends and report macro AUC:
        only Tucker
        only exp169
        average(Tucker, exp169)
        0.7*Tucker + 0.3*exp169
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"
TUCKER_DIR = ROOT / "model-weights" / "tucker_sed"
EXP169_ONNX = CACHE_DIR / "onnx"

sys.path.insert(0, str(ROOT / "experiments" / "_data_pipelines"))
from exp169_distilled_sed import build_primaries, EVAL_SS_N_FILES, SEED  # noqa

SR = 32000
WIN_SAMPLES = SR * 5
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000
TOP_DB = 80


def audio_to_mel(wavs):
    mels = []
    for x in wavs:
        s = librosa.feature.melspectrogram(
            y=x.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def make_session(p):
    return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def predict_5fold(eval_indices, wav_mm, sess_paths, label):
    sessions = [make_session(p) for p in sorted(sess_paths)]
    in_name = sessions[0].get_inputs()[0].name
    n = len(eval_indices)
    out = np.zeros((n, 234), dtype=np.float32)
    batch = 16
    t0 = time.time()
    for i in range(0, n, batch):
        sl = eval_indices[i:i + batch]
        wavs = wav_mm[sl].astype(np.float32)
        mel = audio_to_mel(wavs)
        p_sum = np.zeros((len(sl), 234), dtype=np.float32)
        for s in sessions:
            outs = s.run(None, {in_name: mel})
            clip_logits = outs[0]
            frame_max = outs[1].max(axis=1)
            p_sum += 0.5 * (1 / (1 + np.exp(-np.clip(clip_logits, -50, 50)))) \
                   + 0.5 * (1 / (1 + np.exp(-np.clip(frame_max, -50, 50))))
        out[i:i + len(sl)] = p_sum / len(sessions)
    print(f"  {label} 5-fold predict {time.time()-t0:.0f}s")
    return out


def per_taxon_auc(preds, Y, taxon_idx):
    aucs = np.full(234, np.nan)
    for c in range(234):
        s = Y[:, c].sum()
        if s == 0 or s == len(Y):
            continue
        try:
            aucs[c] = roc_auc_score(Y[:, c], preds[:, c])
        except Exception:
            pass
    rows = []
    for t, idxs in taxon_idx.items():
        if not idxs: continue
        ts = np.array(idxs)
        valid = ~np.isnan(aucs[ts])
        if valid.sum() == 0:
            rows.append((t, len(idxs), 0, np.nan))
            continue
        rows.append((t, len(idxs), int(valid.sum()), float(np.nanmean(aucs[ts[valid]]))))
    return rows, aucs


def main():
    primary, l2i = build_primaries()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    cls_to_taxon = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    taxon_idx = {t: [l2i[c] for c in cls_to_taxon if cls_to_taxon[c] == t and c in l2i]
                 for t in tax["class_name"].unique()}

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
    Y = np.zeros((len(eval_indices), 234), dtype=np.uint8)
    for i, gi in enumerate(eval_indices):
        s = ss_label_str[gi]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        for lbl in str(s).split(";"):
            lbl = lbl.strip()
            if lbl in l2i:
                Y[i, l2i[lbl]] = 1
    print(f"eval rows: {len(eval_indices)}")

    p_tucker = predict_5fold(eval_indices, wav_mm, TUCKER_DIR.glob("sed_fold*.onnx"), "Tucker")
    p_exp169 = predict_5fold(eval_indices, wav_mm, EXP169_ONNX.glob("sed_fold*.onnx"), "exp169")

    # Per-taxon AUC
    rows_t, aucs_t = per_taxon_auc(p_tucker, Y, taxon_idx)
    rows_e, aucs_e = per_taxon_auc(p_exp169, Y, taxon_idx)

    print("\n=== Per-taxon AUC: 5-fold ensembles ===")
    print(f"{'taxon':12s} {'n_cls':>5s} {'n_eval':>6s} {'Tucker':>8s} {'exp169':>8s} {'gap':>7s}")
    rows_t.sort(key=lambda r: -r[1])
    tk = {r[0]: r for r in rows_t}; ek = {r[0]: r for r in rows_e}
    for t, nc, ne, _ in rows_t:
        ta = tk[t][3]; ea = ek[t][3]
        gap = ea - ta if (np.isfinite(ta) and np.isfinite(ea)) else np.nan
        print(f"{t:12s} {nc:>5d} {ne:>6d} {ta:>8.4f} {ea:>8.4f} {gap:>+7.4f}")
    print(f"  macro Tucker  : {np.nanmean(aucs_t):.4f}")
    print(f"  macro exp169  : {np.nanmean(aucs_e):.4f}")

    # Pearson on per-class predictions (with positive variance both sides)
    pcols = []
    for c in range(234):
        if p_tucker[:, c].std() == 0 or p_exp169[:, c].std() == 0:
            continue
        try:
            r, _ = pearsonr(p_tucker[:, c], p_exp169[:, c])
            if np.isfinite(r):
                pcols.append(r)
        except Exception:
            pass
    print(f"\n  per-class Pearson(exp169_5fold, Tucker_5fold):")
    print(f"    mean   = {np.mean(pcols):.4f}")
    print(f"    median = {np.median(pcols):.4f}")
    flat_r, _ = pearsonr(p_tucker.flatten(), p_exp169.flatten())
    print(f"  flat Pearson = {flat_r:.4f}")

    # Blends
    print("\n=== Blend simulations (macro AUC, skip empty) ===")
    blends = {
        "Tucker only           ": p_tucker,
        "exp169 only           ": p_exp169,
        "0.5 Tucker + 0.5 exp169": 0.5 * p_tucker + 0.5 * p_exp169,
        "0.7 Tucker + 0.3 exp169": 0.7 * p_tucker + 0.3 * p_exp169,
        "0.8 Tucker + 0.2 exp169": 0.8 * p_tucker + 0.2 * p_exp169,
    }
    for name, preds in blends.items():
        _, aucs = per_taxon_auc(preds, Y, taxon_idx)
        macro = float(np.nanmean(aucs))
        print(f"  {name}: {macro:.4f}")

    # Save
    out = {
        "n_eval_rows": int(len(eval_indices)),
        "macro_Tucker_5fold": float(np.nanmean(aucs_t)),
        "macro_exp169_5fold": float(np.nanmean(aucs_e)),
        "per_taxon": [
            {"taxon": t, "n_cls": tk[t][1], "n_eval": tk[t][2],
             "tucker_auc": float(tk[t][3]) if np.isfinite(tk[t][3]) else None,
             "exp169_auc": float(ek[t][3]) if np.isfinite(ek[t][3]) else None}
            for t, _, _, _ in rows_t
        ],
        "pearson_per_class_mean": float(np.mean(pcols)),
        "pearson_per_class_median": float(np.median(pcols)),
        "pearson_flat": float(flat_r),
        "blends": {name: float(np.nanmean(per_taxon_auc(p, Y, taxon_idx)[1]))
                   for name, p in blends.items()},
    }
    out_dir = ROOT / "experiments" / "_audits_post_v26" / "exp169_outputs"
    (out_dir / "vs_tucker_5fold.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_dir / 'vs_tucker_5fold.json'}")


if __name__ == "__main__":
    main()

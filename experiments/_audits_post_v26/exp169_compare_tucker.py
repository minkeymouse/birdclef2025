#!/usr/bin/env python3
"""Compare exp169 fold 0 vs Tucker 5-fold per-taxon, plus prediction Pearson.

This answers two questions for the LB-submission go/no-go decision:
  1. Are exp169 and Tucker decorrelated enough that an ensemble adds value?
  2. Are there taxa where exp169 is materially weaker (Insecta is a known
     non-Aves weak point because Perch is bird-only)?
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"
TUCKER_DIR = ROOT / "model-weights" / "tucker_sed"

sys.path.insert(0, str(ROOT / "experiments" / "_data_pipelines"))
from exp169_distilled_sed import DistilledSED, build_primaries, EVAL_SS_N_FILES, SEED  # noqa


SR = 32000
WIN_SAMPLES = SR * 5
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000
TOP_DB = 80


def audio_to_mel(wavs):
    """Match Tucker's preprocessing exactly: per-spec scalar z-score."""
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


def tucker_predict(eval_indices, wav_mm):
    paths = sorted(TUCKER_DIR.glob("sed_fold*.onnx"))
    sessions = [make_session(p) for p in paths]
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
    print(f"  Tucker 5-fold predict {time.time()-t0:.0f}s")
    return out


@torch.no_grad()
def exp169_predict(eval_indices, wav_mm, fold=0):
    state = torch.load(CACHE_DIR / f"fold{fold}" / "best_ckpt.pt", map_location="cpu")
    m = DistilledSED(n_cls=234)
    m.load_state_dict(state["state_dict"])
    m.eval()
    n = len(eval_indices)
    out = np.zeros((n, 234), dtype=np.float32)
    batch = 16
    t0 = time.time()
    for i in range(0, n, batch):
        sl = eval_indices[i:i + batch]
        wavs = wav_mm[sl].astype(np.float32)
        x = torch.from_numpy(wavs)
        clip_l, _, _ = m(x)
        out[i:i + len(sl)] = torch.sigmoid(clip_l).numpy()
    print(f"  exp169 fold{fold} predict {time.time()-t0:.0f}s")
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
        if not idxs:
            continue
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

    print("\nTucker 5-fold predict...")
    p_tucker = tucker_predict(eval_indices, wav_mm)
    print("exp169 fold 0 predict...")
    p_exp169 = exp169_predict(eval_indices, wav_mm, fold=0)

    rows_tucker, aucs_tucker = per_taxon_auc(p_tucker, Y, taxon_idx)
    rows_exp169, aucs_exp169 = per_taxon_auc(p_exp169, Y, taxon_idx)

    print("\n=== Per-taxon AUC comparison ===")
    print(f"{'taxon':12s} {'n_cls':>5s} {'n_eval':>6s} {'Tucker':>8s} {'exp169':>8s} {'gap':>7s}")
    rows_tucker.sort(key=lambda r: -r[1])
    tk = {r[0]: r for r in rows_tucker}
    ek = {r[0]: r for r in rows_exp169}
    for t, nc, ne, _ in rows_tucker:
        ta = tk[t][3]
        ea = ek[t][3]
        gap = ea - ta
        print(f"{t:12s} {nc:>5d} {ne:>6d} {ta:>8.4f} {ea:>8.4f} {gap:>+7.4f}")

    print(f"\n  macro Tucker  : {np.nanmean(aucs_tucker):.4f}")
    print(f"  macro exp169  : {np.nanmean(aucs_exp169):.4f}")

    # Pearson correlation between the two streams (per-class then averaged)
    pcols = []
    for c in range(234):
        if Y[:, c].sum() == 0 and (p_tucker[:, c].std() > 0 or p_exp169[:, c].std() > 0):
            continue
        if p_tucker[:, c].std() == 0 or p_exp169[:, c].std() == 0:
            continue
        try:
            r, _ = pearsonr(p_tucker[:, c], p_exp169[:, c])
            if np.isfinite(r):
                pcols.append(r)
        except Exception:
            pass
    print(f"\n  per-class Pearson(exp169, Tucker) on labeled SS:")
    print(f"    mean   = {np.mean(pcols):.4f}")
    print(f"    median = {np.median(pcols):.4f}")
    print(f"    n_classes_with_signal = {len(pcols)}")
    print(f"  Lower Pearson => more decorrelated => bigger ensemble gain.")

    # Per-class flat (across all 234 × N) scalar Pearson
    flat_r, _ = pearsonr(p_tucker.flatten(), p_exp169.flatten())
    print(f"  flat Pearson(exp169, Tucker) = {flat_r:.4f}")

    # Save for paper / memory
    out = {
        "n_eval_rows": int(len(eval_indices)),
        "macro_Tucker": float(np.nanmean(aucs_tucker)),
        "macro_exp169_fold0": float(np.nanmean(aucs_exp169)),
        "per_taxon": [
            {"taxon": t, "n_cls": tk[t][1], "n_eval": tk[t][2],
             "tucker_mean_auc": float(tk[t][3]) if np.isfinite(tk[t][3]) else None,
             "exp169_mean_auc": float(ek[t][3]) if np.isfinite(ek[t][3]) else None,
             "gap": (float(ek[t][3] - tk[t][3]) if np.isfinite(tk[t][3]) and np.isfinite(ek[t][3]) else None)}
            for t, _, _, _ in rows_tucker
        ],
        "pearson_per_class_mean": float(np.mean(pcols)),
        "pearson_per_class_median": float(np.median(pcols)),
        "pearson_flat": float(flat_r),
    }
    out_dir = ROOT / "experiments" / "_audits_post_v26" / "exp169_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vs_tucker.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_dir / 'vs_tucker.json'}")


if __name__ == "__main__":
    main()

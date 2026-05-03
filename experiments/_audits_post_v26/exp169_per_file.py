#!/usr/bin/env python3
"""Per-file diagnostic of exp169 5-fold on labeled-SS held-out (11 files).

Want to see: are some files much harder than others? Are the worst files
non-Aves dominated? Diagnostic for "data limit" vs "model limit" debate.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
import librosa
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"
EXP169_ONNX = CACHE_DIR / "onnx"
TUCKER_DIR = ROOT / "model-weights" / "tucker_sed"

sys.path.insert(0, str(ROOT / "experiments" / "_data_pipelines"))
from exp169_distilled_sed import build_primaries, EVAL_SS_N_FILES, SEED  # noqa

SR = 32000
N_MELS = 256; N_FFT = 2048; HOP = 512; FMIN = 20; FMAX = 16000; TOP_DB = 80


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


def predict_5fold(indices, wav_mm, paths):
    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in sorted(paths)]
    in_name = sessions[0].get_inputs()[0].name
    out = np.zeros((len(indices), 234), dtype=np.float32)
    for i in range(0, len(indices), 16):
        sl = indices[i:i + 16]
        wavs = wav_mm[sl].astype(np.float32)
        mel = audio_to_mel(wavs)
        p_sum = np.zeros((len(sl), 234), dtype=np.float32)
        for s in sessions:
            outs = s.run(None, {in_name: mel})
            cl = outs[0]; fm = outs[1].max(axis=1)
            p_sum += 0.5 * (1 / (1 + np.exp(-np.clip(cl, -50, 50)))) \
                   + 0.5 * (1 / (1 + np.exp(-np.clip(fm, -50, 50))))
        out[i:i + len(sl)] = p_sum / len(sessions)
    return out


def main():
    primary, l2i = build_primaries()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    cls_to_taxon = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))

    wav_mm = np.load(CACHE_DIR / "waveforms_fp16.npy", mmap_mode="r")
    meta = np.load(CACHE_DIR / "meta.npz", allow_pickle=True)
    filenames = meta["filenames"]
    is_ss = meta["is_ss"]
    ss_label_str = meta["ss_label_str"]

    ss_files = sorted({filenames[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(SEED); rng.shuffle(ss_files)
    eval_files = list(ss_files[:EVAL_SS_N_FILES])
    eval_indices = np.array(
        [i for i in np.where(is_ss == 1)[0] if filenames[i] in eval_files],
        dtype=np.int64,
    )

    Y = np.zeros((len(eval_indices), 234), dtype=np.uint8)
    file_of_row = np.empty(len(eval_indices), dtype=object)
    for i, gi in enumerate(eval_indices):
        s = ss_label_str[gi]
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            for lbl in str(s).split(";"):
                lbl = lbl.strip()
                if lbl in l2i:
                    Y[i, l2i[lbl]] = 1
        file_of_row[i] = filenames[gi]

    p_t = predict_5fold(eval_indices, wav_mm, TUCKER_DIR.glob("sed_fold*.onnx"))
    p_e = predict_5fold(eval_indices, wav_mm, EXP169_ONNX.glob("sed_fold*.onnx"))

    print(f"\n=== Per-file macro AUC ===")
    print(f"{'file':50s}  {'rows':>4s}  {'pos_cls':>7s}  {'Tucker':>8s}  {'exp169':>8s}  {'pos_taxa'}")
    for f in eval_files:
        m = file_of_row == f
        if m.sum() == 0:
            continue
        p_t_f = p_t[m]; p_e_f = p_e[m]; Y_f = Y[m]
        # macro per file
        aucs_t, aucs_e = [], []
        pos_cls = (Y_f.sum(axis=0) > 0).sum()
        for c in range(234):
            s = Y_f[:, c].sum()
            if s == 0 or s == m.sum():
                continue
            try:
                aucs_t.append(roc_auc_score(Y_f[:, c], p_t_f[:, c]))
                aucs_e.append(roc_auc_score(Y_f[:, c], p_e_f[:, c]))
            except Exception:
                pass
        # classes present in this file's positives (any row)
        present_cls = np.where(Y_f.any(axis=0))[0]
        taxa = sorted({cls_to_taxon.get(primary[c], "?") for c in present_cls})
        m_t = float(np.mean(aucs_t)) if aucs_t else float('nan')
        m_e = float(np.mean(aucs_e)) if aucs_e else float('nan')
        print(f"{f:50s}  {m.sum():>4d}  {pos_cls:>7d}  {m_t:>8.4f}  {m_e:>8.4f}  {taxa}")


if __name__ == "__main__":
    main()

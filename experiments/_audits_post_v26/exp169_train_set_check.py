#!/usr/bin/env python3
"""Sanity: does exp169 also score 0.99+ on labeled SS rows it actually trained on?

Disambiguates whether the Tucker vs exp169 gap on the 11 held-out files is:
  (a) Tucker memorising labeled SS (which it trained on)
  (b) exp169 being genuinely weaker
By running exp169 5-fold ensemble on the 55 SS-train files (which exp169 saw),
we measure the exp169 memorisation ceiling. If close to 0.99, hypothesis (a)
dominates; if it stays at 0.84, hypothesis (b) dominates.
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

SR = 32000; N_MELS = 256; N_FFT = 2048; HOP = 512
FMIN = 20; FMAX = 16000; TOP_DB = 80


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


def macro_auc(p, Y):
    aucs = []
    for c in range(234):
        s = Y[:, c].sum()
        if s == 0 or s == len(Y):
            continue
        try:
            aucs.append(roc_auc_score(Y[:, c], p[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


def main():
    primary, l2i = build_primaries()
    wav_mm = np.load(CACHE_DIR / "waveforms_fp16.npy", mmap_mode="r")
    meta = np.load(CACHE_DIR / "meta.npz", allow_pickle=True)
    filenames = meta["filenames"]
    is_ss = meta["is_ss"]
    ss_label_str = meta["ss_label_str"]
    ss_files = sorted({filenames[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(SEED); rng.shuffle(ss_files)
    eval_files = set(ss_files[:EVAL_SS_N_FILES])

    train_indices = np.array(
        [i for i in np.where(is_ss == 1)[0] if filenames[i] not in eval_files],
        dtype=np.int64,
    )
    eval_indices = np.array(
        [i for i in np.where(is_ss == 1)[0] if filenames[i] in eval_files],
        dtype=np.int64,
    )
    print(f"SS rows  train (exp169 SAW): {len(train_indices)}  eval (exp169 held out): {len(eval_indices)}")

    def build_Y(indices):
        Y = np.zeros((len(indices), 234), dtype=np.uint8)
        for i, gi in enumerate(indices):
            s = ss_label_str[gi]
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            for lbl in str(s).split(";"):
                lbl = lbl.strip()
                if lbl in l2i:
                    Y[i, l2i[lbl]] = 1
        return Y

    Y_train = build_Y(train_indices)
    Y_eval = build_Y(eval_indices)

    print("\nexp169 5-fold predictions...")
    p_train_e = predict_5fold(train_indices, wav_mm, EXP169_ONNX.glob("sed_fold*.onnx"))
    p_eval_e = predict_5fold(eval_indices, wav_mm, EXP169_ONNX.glob("sed_fold*.onnx"))
    print("Tucker 5-fold predictions...")
    p_train_t = predict_5fold(train_indices, wav_mm, TUCKER_DIR.glob("sed_fold*.onnx"))
    p_eval_t = predict_5fold(eval_indices, wav_mm, TUCKER_DIR.glob("sed_fold*.onnx"))

    print("\n=== labeled SS macro AUC by row partition ===")
    print(f"  partition           rows   Tucker_macro   exp169_macro")
    for label, p_t, p_e, Y in [
        ("train (both saw)  ", p_train_t, p_train_e, Y_train),
        ("eval  (Tucker saw)", p_eval_t,  p_eval_e,  Y_eval),
    ]:
        m_t, n_t = macro_auc(p_t, Y)
        m_e, n_e = macro_auc(p_e, Y)
        print(f"  {label}  {len(Y):>5d}    {m_t:.4f} ({n_t:>3d})   {m_e:.4f} ({n_e:>3d})")

    # Memorisation ratio: train_macro / eval_macro
    print("\nIf exp169 train_macro >> eval_macro, the model can memorise — gap on")
    print("eval is an out-of-distribution / file generalisation gap.")
    print("If train_macro ~ eval_macro, exp169 simply doesn't memorise — recipe limit.")


if __name__ == "__main__":
    main()

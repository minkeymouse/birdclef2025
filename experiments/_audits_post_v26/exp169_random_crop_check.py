#!/usr/bin/env python3
"""Lower-bound the random-crop gap.

We took the center-5s of every file. Inference re-extracts the SAME center
5s. So at inference time we evaluate on the exact slice the model trained
on for labeled-SS-train rows. That's a "perfect" condition for the
center-crop recipe and reaches 0.9971 on those rows.

What if we eval the same fold-0 ckpt on RANDOM 5s crops drawn from the
same labeled-SS-train files? If the model only learned the center
fingerprint, AUC on random crops will collapse. If the model learned
the species' acoustic content, AUC on random crops will stay high.

This gives a lower bound on what we lose by not training on random
crops — if even our centre-crop-trained model handles random crops
fine, then random crops at training time are not the bottleneck.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
CACHE_DIR = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"

sys.path.insert(0, str(ROOT / "experiments" / "_data_pipelines"))
from exp169_distilled_sed import DistilledSED, build_primaries, EVAL_SS_N_FILES, SEED  # noqa

SR = 32000
WIN_SAMPLES = SR * 5
FILE_SAMPLES = SR * 60


def load_ss_random_crop(path: Path, end_sec: int, rng: np.random.RandomState) -> np.ndarray:
    """Load 5s window for the labeled SS row at end_sec, but jitter ±2 sec."""
    try:
        wav, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != SR:
            import torchaudio.functional as TF
            wav = TF.resample(torch.from_numpy(wav), sr0, SR).numpy()
        # the labeled SS window is exact 5-sec [end_sec-5, end_sec)
        # jitter the start by up to ±2 sec, clipped to file bounds
        s_center = (end_sec - 5) * SR
        jitter = rng.randint(-2 * SR, 2 * SR + 1)
        s = max(0, min(len(wav) - WIN_SAMPLES, s_center + jitter))
        clip = wav[s:s + WIN_SAMPLES]
        if len(clip) < WIN_SAMPLES:
            clip = np.pad(clip, (0, WIN_SAMPLES - len(clip)))
        return clip.astype(np.float32)
    except Exception:
        return np.zeros(WIN_SAMPLES, dtype=np.float32)


@torch.no_grad()
def predict(model, wavs: np.ndarray, dev: str = "cpu") -> np.ndarray:
    out = np.zeros((len(wavs), 234), dtype=np.float32)
    for i in range(0, len(wavs), 16):
        x = torch.from_numpy(wavs[i:i + 16]).to(dev)
        clip_l, _, _ = model(x)
        out[i:i + 16] = torch.sigmoid(clip_l).cpu().numpy()
    return out


def main():
    primary, l2i = build_primaries()
    state = torch.load(CACHE_DIR / "fold0" / "best_ckpt.pt", map_location="cpu")
    m = DistilledSED(n_cls=234)
    m.load_state_dict(state["state_dict"])
    m.eval()

    # Cached center crops for SS train rows, AND random-crop versions
    meta = np.load(CACHE_DIR / "meta.npz", allow_pickle=True)
    filenames = meta["filenames"]
    is_ss = meta["is_ss"]
    ss_endsec = meta["ss_endsec"]
    ss_label_str = meta["ss_label_str"]
    wav_mm = np.load(CACHE_DIR / "waveforms_fp16.npy", mmap_mode="r")

    ss_files = sorted({filenames[i] for i in np.where(is_ss == 1)[0]})
    rng = np.random.RandomState(SEED); rng.shuffle(ss_files)
    eval_files = set(ss_files[:EVAL_SS_N_FILES])

    # Pick 200 random rows from SS train for the comparison
    ss_train_idx = np.array(
        [i for i in np.where(is_ss == 1)[0] if filenames[i] not in eval_files],
        dtype=np.int64,
    )
    rng_eval = np.random.RandomState(0)
    sub = rng_eval.choice(ss_train_idx, size=min(200, len(ss_train_idx)), replace=False)

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

    Y = build_Y(sub)

    # 1) center crop (cache hit — exact training input)
    wavs_center = wav_mm[sub].astype(np.float32)
    p_center = predict(m, wavs_center)

    # 2) random crop (jittered ±2 sec)
    rng_jit = np.random.RandomState(7)
    wavs_random = []
    for gi in sub:
        wavs_random.append(load_ss_random_crop(
            DATA / "train_soundscapes" / filenames[gi], int(ss_endsec[gi]), rng_jit
        ))
    wavs_random = np.stack(wavs_random)
    p_random = predict(m, wavs_random)

    def macro(p, Y):
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

    m_c, n_c = macro(p_center, Y)
    m_r, n_r = macro(p_random, Y)
    print(f"\nfold-0 ckpt on {len(sub)} labeled-SS-train rows:")
    print(f"  center crop  (training input): macro AUC = {m_c:.4f}  ({n_c} cls)")
    print(f"  random crop  (jittered ±2s):  macro AUC = {m_r:.4f}  ({n_r} cls)")
    print(f"  drop = {m_c - m_r:.4f}")
    print()
    print("Interpretation:")
    print("  Small drop ⇒ model is robust to within-file shift; centre crop")
    print("    cache is *not* the gap source — recipe is fundamentally weaker.")
    print("  Large drop ⇒ model overfit on centre 5s position; random-crop")
    print("    training would help substantially.")


if __name__ == "__main__":
    main()

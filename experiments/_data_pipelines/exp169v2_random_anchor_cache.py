#!/usr/bin/env python3
"""exp169 v2 — pre-compute Perch embeddings at K random anchors per file.

Difference vs exp169 v1: instead of one centre-5s slice per file, we sample
K=3 anchors per file (or fewer if file is too short). Each anchor stores
both its starting offset (in samples) and the Perch embedding of the
5-second window starting there. The waveform itself is *not* cached
(too large for K>1); the training loader decodes on-the-fly.

Outputs:
  experiments/_data_pipelines/exp169v2_outputs/anchors.npz
    files       : (N_files,)        object   — relative path under train_audio/
    is_ss       : (N_files,)        uint8
    ss_endsec   : (N_files,)        int32    — labeled-SS row's window end
    primary     : (N_files,)        int32
    n_anchors   : (N_files,)        int8     — actual anchor count for that file
    anchor_off  : (N_files, K)      int32    — sample offset (-1 = unused)
    anchor_emb  : (N_files, K, 1536) float32 — Perch teacher embedding
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
PERCH_ONNX = ROOT / "model-weights" / "perch_v2_onnx" / "perch_v2.onnx"
OUT = ROOT / "experiments" / "_data_pipelines" / "exp169v2_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC
K = 3                    # anchors per file
RANDOM_SEED = 42
ONNX_BATCH = 32


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def pick_anchors(wav_len: int, k: int, rng: np.random.RandomState) -> list[int]:
    """Pick up to k random non-overlapping start offsets (samples)."""
    if wav_len <= WIN_SAMPLES:
        return [0]
    max_start = wav_len - WIN_SAMPLES
    # uniformly spaced + jitter in [0, gap/2)
    n = min(k, max(1, max_start // (WIN_SAMPLES // 2)))
    bases = np.linspace(0, max_start, n)
    offs = []
    for b in bases:
        jitter = rng.randint(0, max(1, int(b + WIN_SAMPLES) - int(b)))
        s = int(b + jitter * 0.0)  # keep simple — base only
        offs.append(int(min(max_start, max(0, s))))
    # add a fully random extra anchor if k allows
    if n < k:
        offs.append(int(rng.randint(0, max_start + 1)))
    return offs[:k]


def load_audio_mono(path: Path) -> np.ndarray | None:
    try:
        wav, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != SR:
            import torchaudio.functional as TF
            import torch
            wav = TF.resample(torch.from_numpy(wav), sr0, SR).numpy()
        return wav.astype(np.float32)
    except Exception as exc:
        print(f"  audio fail {path.name}: {exc}", flush=True)
        return None


def main():
    primary, l2i = build_primaries()
    ta = pd.read_csv(DATA / "train.csv")
    ta = ta[ta["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    ta["primary_idx"] = ta["primary_label"].astype(str).map(l2i)

    ss_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    ss_raw["end_sec"] = pd.to_timedelta(ss_raw["end"]).dt.total_seconds().astype(int)
    ss_g = (ss_raw.groupby(["filename", "end_sec"])["primary_label"]
            .apply(lambda s: ";".join(sorted({l for x in s for l in str(x).split(";") if l.strip()}))).reset_index())

    n = len(ta) + len(ss_g)
    files = np.empty(n, dtype=object)
    is_ss = np.zeros(n, dtype=np.uint8)
    ss_endsec = np.full(n, -1, dtype=np.int32)
    primary_idx = np.full(n, -1, dtype=np.int32)
    ss_label_str = np.empty(n, dtype=object)
    n_anchors = np.zeros(n, dtype=np.int8)
    anchor_off = np.full((n, K), -1, dtype=np.int32)
    anchor_emb = np.zeros((n, K, 1536), dtype=np.float32)

    rng = np.random.RandomState(RANDOM_SEED)
    sess = ort.InferenceSession(
        str(PERCH_ONNX),
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    print(f"Perch ONNX  provider={sess.get_providers()[0]}  total files: {n}", flush=True)

    # We'll build a flat list of (row_index, anchor_index, waveform) and
    # batch through Perch in ONNX_BATCH chunks for throughput.
    pending_wavs: list[np.ndarray] = []
    pending_keys: list[tuple[int, int]] = []  # (row_idx, anchor_idx)

    def flush_batch():
        if not pending_wavs:
            return
        x = np.stack(pending_wavs).astype(np.float32)
        out = sess.run(None, {in_name: x})
        emb = out[0]
        for k_idx, (ri, ai) in enumerate(pending_keys):
            anchor_emb[ri, ai] = emb[k_idx]
        pending_wavs.clear(); pending_keys.clear()

    t0 = time.time()
    # TA rows
    for ri, row in enumerate(ta.itertuples()):
        files[ri] = row.filename
        primary_idx[ri] = int(row.primary_idx)
        wav = load_audio_mono(DATA / "train_audio" / row.filename)
        if wav is None or len(wav) == 0:
            offs = [0]
            wavs_at = [np.zeros(WIN_SAMPLES, dtype=np.float32)]
        else:
            if len(wav) < WIN_SAMPLES:
                reps = WIN_SAMPLES // len(wav) + 1
                wav = np.tile(wav, reps)[:WIN_SAMPLES]
                offs = [0]
                wavs_at = [wav.astype(np.float32)]
            else:
                offs = pick_anchors(len(wav), K, rng)
                wavs_at = [wav[o:o + WIN_SAMPLES].astype(np.float32) for o in offs]
        n_anchors[ri] = len(offs)
        for ai, (off, w) in enumerate(zip(offs, wavs_at)):
            anchor_off[ri, ai] = int(off)
            pending_wavs.append(w)
            pending_keys.append((ri, ai))
            if len(pending_wavs) >= ONNX_BATCH:
                flush_batch()
        if (ri + 1) % 2000 == 0:
            print(f"  TA {ri+1}/{len(ta)}  elapsed {time.time()-t0:.0f}s", flush=True)

    base = len(ta)
    for j, row in enumerate(ss_g.itertuples()):
        ri = base + j
        files[ri] = row.filename
        is_ss[ri] = 1
        ss_endsec[ri] = int(row.end_sec)
        ss_label_str[ri] = row.primary_label
        wav = load_audio_mono(DATA / "train_soundscapes" / row.filename)
        if wav is None or len(wav) == 0:
            offs = [0]
            wavs_at = [np.zeros(WIN_SAMPLES, dtype=np.float32)]
        else:
            # for labeled SS the "correct" 5-sec window is [end_sec-5, end_sec)
            # so fix the first anchor to that window, optional jitter for other anchors
            base_start = max(0, (int(row.end_sec) - WIN_SEC) * SR)
            base_start = min(base_start, len(wav) - WIN_SAMPLES) if len(wav) > WIN_SAMPLES else 0
            offs = [base_start]
            for _ in range(K - 1):
                lo = max(0, base_start - SR)
                hi = min(len(wav) - WIN_SAMPLES, base_start + SR) if len(wav) > WIN_SAMPLES else 0
                if hi > lo:
                    offs.append(int(rng.randint(lo, hi + 1)))
                else:
                    offs.append(base_start)
            offs = offs[:K]
            wavs_at = []
            for o in offs:
                clip = wav[o:o + WIN_SAMPLES]
                if len(clip) < WIN_SAMPLES:
                    clip = np.pad(clip, (0, WIN_SAMPLES - len(clip)))
                wavs_at.append(clip.astype(np.float32))
        n_anchors[ri] = len(offs)
        for ai, (off, w) in enumerate(zip(offs, wavs_at)):
            anchor_off[ri, ai] = int(off)
            pending_wavs.append(w)
            pending_keys.append((ri, ai))
            if len(pending_wavs) >= ONNX_BATCH:
                flush_batch()
        if (j + 1) % 200 == 0:
            print(f"  SS {j+1}/{len(ss_g)}  elapsed {time.time()-t0:.0f}s", flush=True)
    flush_batch()
    print(f"all forwards done in {time.time()-t0:.0f}s", flush=True)

    np.savez(
        OUT / "anchors.npz",
        files=files, is_ss=is_ss, ss_endsec=ss_endsec, primary_idx=primary_idx,
        ss_label_str=np.asarray(ss_label_str), n_anchors=n_anchors,
        anchor_off=anchor_off, anchor_emb=anchor_emb,
    )
    sz = (OUT / "anchors.npz").stat().st_size / 1024 / 1024
    print(f"saved anchors.npz {sz:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

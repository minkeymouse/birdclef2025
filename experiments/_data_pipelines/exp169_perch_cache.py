#!/usr/bin/env python3
"""exp169 — pre-cache Perch v2 embeddings for distillation training.

Strategy: extract one 5-second waveform per file (center crop) and run it
through Perch v2 ONNX (CUDA). Save (waveform, embedding) pairs to a single
.npz so the training dataloader can stream without ONNX in the loop.

Output:
  experiments/_data_pipelines/exp169_outputs/perch_cache.npz
    waveforms : (N, 160000) float16  — center 5s @ 32kHz, fp16 to save disk
    perch_emb : (N, 1536)   float32
    filenames : (N,)        object   — relative path under train_audio/
    is_ss     : (N,)        uint8    — 0 = train_audio, 1 = labeled SS
    ss_endsec : (N,)        int32    — only meaningful when is_ss==1
    primary   : (N,)        int32    — primary class index, -1 for SS rows

Usage: uv run python experiments/_data_pipelines/exp169_perch_cache.py
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
OUT = ROOT / "experiments" / "_data_pipelines" / "exp169_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC  # 160000
BATCH = 32


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def load_audio_center5s(path: Path) -> np.ndarray:
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            import torchaudio.functional as TF
            import torch
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
        if len(wav) == 0:
            return np.zeros(WIN_SAMPLES, dtype=np.float32)
        if len(wav) < WIN_SAMPLES:
            reps = WIN_SAMPLES // len(wav) + 1
            wav = np.tile(wav, reps)[:WIN_SAMPLES]
            return wav.astype(np.float32)
        s = (len(wav) - WIN_SAMPLES) // 2
        return wav[s:s + WIN_SAMPLES].astype(np.float32)
    except Exception as exc:
        print(f"  audio fail {path.name}: {exc}", flush=True)
        return np.zeros(WIN_SAMPLES, dtype=np.float32)


def load_ss_window(path: Path, end_sec: int) -> np.ndarray:
    """Load the 5-sec window ending at `end_sec` from a labeled SS file."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            import torchaudio.functional as TF
            import torch
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
        start_samp = max(0, (end_sec - WIN_SEC) * SR)
        end_samp = start_samp + WIN_SAMPLES
        clip = wav[start_samp:end_samp]
        if len(clip) < WIN_SAMPLES:
            clip = np.pad(clip, (0, WIN_SAMPLES - len(clip)))
        return clip.astype(np.float32)
    except Exception as exc:
        print(f"  ss fail {path.name} end={end_sec}: {exc}", flush=True)
        return np.zeros(WIN_SAMPLES, dtype=np.float32)


def main():
    primary, l2i = build_primaries()
    ta = pd.read_csv(DATA / "train.csv")
    ta = ta[ta["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    ta["primary_idx"] = ta["primary_label"].astype(str).map(l2i)
    print(f"train_audio rows: {len(ta)}")

    ss_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    ss_raw["end_sec"] = pd.to_timedelta(ss_raw["end"]).dt.total_seconds().astype(int)
    ss_g = (ss_raw.groupby(["filename", "end_sec"])["primary_label"]
            .apply(lambda s: ";".join(sorted({l for x in s for l in str(x).split(";") if l.strip()}))).reset_index())
    print(f"labeled SS windows: {len(ss_g)}")

    total = len(ta) + len(ss_g)
    print(f"total clips to embed: {total}")

    # Pre-allocate
    waveforms = np.zeros((total, WIN_SAMPLES), dtype=np.float16)
    perch_emb = np.zeros((total, 1536), dtype=np.float32)
    filenames = np.empty(total, dtype=object)
    is_ss = np.zeros(total, dtype=np.uint8)
    ss_endsec = np.zeros(total, dtype=np.int32)
    primary_idx = np.full(total, -1, dtype=np.int32)
    ss_label_str = np.empty(total, dtype=object)  # ";"-joined labels for SS rows

    # Decode all audio first (fastest with sequential, but we can parallelise via batches)
    print("decoding audio...", flush=True)
    t0 = time.time()
    for i, row in enumerate(ta.itertuples()):
        wav = load_audio_center5s(DATA / "train_audio" / row.filename)
        waveforms[i] = wav.astype(np.float16)
        filenames[i] = row.filename
        primary_idx[i] = int(row.primary_idx)
        if (i + 1) % 2000 == 0:
            print(f"  TA {i+1}/{len(ta)}  ({time.time()-t0:.0f}s)", flush=True)
    base = len(ta)
    for j, row in enumerate(ss_g.itertuples()):
        wav = load_ss_window(DATA / "train_soundscapes" / row.filename, int(row.end_sec))
        waveforms[base + j] = wav.astype(np.float16)
        filenames[base + j] = row.filename
        is_ss[base + j] = 1
        ss_endsec[base + j] = int(row.end_sec)
        ss_label_str[base + j] = row.primary_label
        if (j + 1) % 200 == 0:
            print(f"  SS {j+1}/{len(ss_g)}", flush=True)
    print(f"decode done in {time.time()-t0:.0f}s", flush=True)

    # Run Perch ONNX in batches
    print("loading Perch ONNX...", flush=True)
    sess = ort.InferenceSession(
        str(PERCH_ONNX),
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    emb_idx = [i for i, o in enumerate(sess.get_outputs()) if o.name == "embedding"][0]
    print(f"  Perch loaded; provider={sess.get_providers()[0]}  in={in_name}", flush=True)

    t0 = time.time()
    n_batches = (total + BATCH - 1) // BATCH
    for bi in range(n_batches):
        s = bi * BATCH
        e = min(total, s + BATCH)
        x = waveforms[s:e].astype(np.float32)  # ONNX wants fp32
        # Perch expects shape (B, 160000)
        out = sess.run(None, {in_name: x})
        perch_emb[s:e] = out[emb_idx]
        if (bi + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"  batch {bi+1}/{n_batches}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
    print(f"Perch forward done in {time.time()-t0:.0f}s", flush=True)

    # Save: waveforms as uncompressed memmap-friendly .npy (large), metadata in .npz
    print("saving waveforms (uncompressed)...", flush=True)
    np.save(OUT / "waveforms_fp16.npy", waveforms)
    print("saving perch_emb...", flush=True)
    np.save(OUT / "perch_emb.npy", perch_emb)
    print("saving metadata...", flush=True)
    np.savez(
        OUT / "meta.npz",
        filenames=np.asarray(filenames),
        is_ss=is_ss,
        ss_endsec=ss_endsec,
        primary_idx=primary_idx,
        ss_label_str=np.asarray(ss_label_str),
    )
    sz_wav = (OUT / "waveforms_fp16.npy").stat().st_size / 1024 / 1024
    sz_emb = (OUT / "perch_emb.npy").stat().st_size / 1024 / 1024
    print(f"saved waveforms_fp16.npy {sz_wav:.0f} MB, perch_emb.npy {sz_emb:.0f} MB  ({total} clips)", flush=True)


if __name__ == "__main__":
    main()

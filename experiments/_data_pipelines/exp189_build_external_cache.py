#!/usr/bin/env python3
"""exp189 — extend the exp169v2 anchor cache with EXTERNAL non-Aves focal clips.

The winning BirdCLEF-2025 recipe adds external non-Aves data to the SED training set.
The exp169v2 cache (train_audio + labeled SS only) never included data/external. This
builds a NEW cache = exp169v2 anchors + Perch embeddings for external non-Aves clips,
so the SED student sees more non-Aves supervision. Tucker recipe otherwise unchanged
(this STRENGTHENS the SED with data; it does NOT dilute it with a weaker model — the
structural difference from the refuted exp187).

External clips stored with files="../../external/<label>/<file>" + is_ss=0, so the
training loader resolves DATA/train_audio/../../external/... -> data/external/... .
m4a are unreadable by soundfile -> use the pre-converted .ogg (skip .m4a).

Out: experiments/_data_pipelines/exp189_outputs/anchors.npz
"""
from __future__ import annotations
import time, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
EXT = ROOT / "data" / "external"
PERCH_ONNX = ROOT / "model-weights" / "perch_v2_onnx" / "perch_v2.onnx"
SRC_CACHE = ROOT / "experiments" / "_data_pipelines" / "exp169v2_outputs" / "anchors.npz"
OUT = ROOT / "experiments" / "_data_pipelines" / "exp189_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000; WIN_SEC = 5; WIN_SAMPLES = SR * WIN_SEC; K = 3
RANDOM_SEED = 42; ONNX_BATCH = 32


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def pick_anchors(wav_len, k, rng):
    if wav_len <= WIN_SAMPLES:
        return [0]
    max_start = wav_len - WIN_SAMPLES
    n = min(k, max(1, max_start // (WIN_SAMPLES // 2)))
    bases = np.linspace(0, max_start, n)
    offs = [int(min(max_start, max(0, int(b)))) for b in bases]
    if n < k:
        offs.append(int(rng.randint(0, max_start + 1)))
    return offs[:k]


def load_audio_mono(path):
    try:
        wav, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != SR:
            import torchaudio.functional as TF, torch
            wav = TF.resample(torch.from_numpy(wav), sr0, SR).numpy()
        return wav.astype(np.float32)
    except Exception as exc:
        print(f"  audio fail {os.path.basename(str(path))}: {str(exc)[:50]}", flush=True)
        return None


def main():
    primary, l2i = build_primaries()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    nonaves = set(tax[tax.class_name != "Aves"].primary_label.astype(str))

    # enumerate external non-Aves clips: ogg/mp3/wav (m4a -> use converted .ogg, skip .m4a)
    clips = []  # (primary_idx, rel_path_under_train_audio, abs_path)
    for lab in sorted(nonaves):
        d = EXT / lab
        if not d.is_dir() or lab not in l2i:
            continue
        seen_stems = set()
        for f in sorted(glob.glob(str(d / "*"))):
            ext = os.path.splitext(f)[1].lower()
            # skip .mp3 (corrupt headers stall the training dataloader; mp3 stems are ~all dup of ogg/wav
            # -> only 3 clips lost) and .m4a (soundfile can't read; converted twins are .ogg)
            if ext not in (".ogg", ".wav"):
                continue
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem in seen_stems:  # avoid dup if both .ogg and (converted) exist
                continue
            seen_stems.add(stem)
            rel = f"../../external/{lab}/{os.path.basename(f)}"
            clips.append((int(l2i[lab]), rel, f))
    n = len(clips)
    print(f"external non-Aves clips: {n} across {len({c[0] for c in clips})} species", flush=True)

    files = np.empty(n, dtype=object)
    is_ss = np.zeros(n, dtype=np.uint8)
    ss_endsec = np.full(n, -1, dtype=np.int32)
    primary_idx = np.full(n, -1, dtype=np.int32)
    ss_label_str = np.array([""] * n, dtype=object)
    n_anchors = np.zeros(n, dtype=np.int8)
    anchor_off = np.full((n, K), -1, dtype=np.int32)
    anchor_emb = np.zeros((n, K, 1536), dtype=np.float32)

    rng = np.random.RandomState(RANDOM_SEED)
    sess = ort.InferenceSession(str(PERCH_ONNX),
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print(f"Perch ONNX provider={sess.get_providers()[0]}", flush=True)

    pend_w, pend_k = [], []
    def flush():
        if not pend_w:
            return
        out = sess.run(None, {in_name: np.stack(pend_w).astype(np.float32)})[0]
        for ki, (ri, ai) in enumerate(pend_k):
            anchor_emb[ri, ai] = out[ki]
        pend_w.clear(); pend_k.clear()

    t0 = time.time(); n_fail = 0
    for ri, (pidx, rel, ap) in enumerate(clips):
        files[ri] = rel; primary_idx[ri] = pidx
        wav = load_audio_mono(ap)
        if wav is None or len(wav) == 0:
            n_fail += 1
            offs = [0]; wavs = [np.zeros(WIN_SAMPLES, np.float32)]
        elif len(wav) < WIN_SAMPLES:
            reps = WIN_SAMPLES // len(wav) + 1
            wav = np.tile(wav, reps)[:WIN_SAMPLES]
            offs = [0]; wavs = [wav.astype(np.float32)]
        else:
            offs = pick_anchors(len(wav), K, rng)
            wavs = [wav[o:o + WIN_SAMPLES].astype(np.float32) for o in offs]
        n_anchors[ri] = len(offs)
        for ai, (off, w) in enumerate(zip(offs, wavs)):
            anchor_off[ri, ai] = int(off)
            pend_w.append(w); pend_k.append((ri, ai))
            if len(pend_w) >= ONNX_BATCH:
                flush()
        if (ri + 1) % 200 == 0:
            print(f"  {ri+1}/{n}  elapsed {time.time()-t0:.0f}s  fails={n_fail}", flush=True)
    flush()
    print(f"forwards done {time.time()-t0:.0f}s  fails={n_fail}", flush=True)

    # concat with exp169v2 cache
    src = np.load(SRC_CACHE, allow_pickle=True)
    merged = dict(
        files=np.concatenate([src["files"], files]),
        is_ss=np.concatenate([src["is_ss"], is_ss]),
        ss_endsec=np.concatenate([src["ss_endsec"], ss_endsec]),
        primary_idx=np.concatenate([src["primary_idx"], primary_idx]),
        ss_label_str=np.concatenate([src["ss_label_str"], ss_label_str]),
        n_anchors=np.concatenate([src["n_anchors"], n_anchors]),
        anchor_off=np.concatenate([src["anchor_off"], anchor_off]),
        anchor_emb=np.concatenate([src["anchor_emb"], anchor_emb]),
    )
    np.savez(OUT / "anchors.npz", **merged)
    sz = (OUT / "anchors.npz").stat().st_size / 1024 / 1024
    print(f"saved {OUT/'anchors.npz'} {sz:.0f}MB  total_rows={len(merged['files'])} "
          f"(src={len(src['files'])} + ext={n})", flush=True)


if __name__ == "__main__":
    main()

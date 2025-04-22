#!/usr/bin/env python
"""
process.py ‑‑ BirdCLEF 2025 preprocessing pipeline
=================================================
Cleans raw *train_audio* / *train_soundscape* recordings, applies Voice‑Activity
Detection (VAD), deduplicates, builds **10‑second mel‑spectrogram chunks** with
*soft labels* and sample weighting, then saves three artefacts under
``CFG.PROCESSED_DIR``:

* ``mels/<split>/.../*.npy`` ‑ normalised mel spectrogram arrays
* ``labels/<split>/.../*.label.npy`` ‑ dense probability vectors (206‑long)
* ``<split>_metadata.csv`` – one row per chunk with paths + metadata

The script is idempotent and safe to rerun after you add new recordings or
update configuration in ``configure.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import timm

from configure import CFG
from data_utils import (
    compute_mel,
    compute_noise_metric,
    load_audio,
    load_vad,
    remove_speech,
    seed_everything,
    segment_audio,
    trim_silence,
)

# ────────────────────────────────────────────────────────────────────────────────
# Globals derived from configuration / dataset
# ────────────────────────────────────────────────────────────────────────────────

# ── Single‑model benchmark loader ────────────────────────────────────────────────
class BenchmarkModel(torch.nn.Module):
    """
    Wraps an EfficientNet‑B0 to load a single state_dict for benchmark smoothing.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=1,
            num_classes=num_classes,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_benchmark() -> Optional[torch.nn.Module]:
    """
    Load a single‑checkpoint benchmark model (state_dict) and return it in eval mode.
    """
    if not CFG.BENCHMARK_MODEL:
        return None
    bp = Path(CFG.BENCHMARK_MODEL)
    if not bp.exists():
        logging.warning("Benchmark path %s missing – skipping.", bp)
        return None

    ckpt = torch.load(bp, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model = BenchmarkModel(num_classes=len(ALL_CLASSES))
    model.load_state_dict(state)
    model.eval()
    return model


# ── Class list discovery ────────────────────────────────────────────────────────
def _discover_classes() -> List[str]:
    if hasattr(CFG, "CLASSES") and CFG.CLASSES:
        return list(CFG.CLASSES)
    if CFG.TAXONOMY_CSV.exists():
        df_tax = pd.read_csv(CFG.TAXONOMY_CSV)
        if "primary_label" in df_tax.columns:
            return sorted(df_tax["primary_label"].unique())
    if CFG.TRAIN_CSV.exists():
        df_train = pd.read_csv(CFG.TRAIN_CSV)
        if "primary_label" in df_train.columns:
            return sorted(df_train["primary_label"].unique())
    raise RuntimeError("Unable to infer species list – please set CFG.CLASSES")

ALL_CLASSES: List[str] = _discover_classes()
CLASS2IDX: Dict[str, int] = {sp: i for i, sp in enumerate(ALL_CLASSES)}

# weight hyper‑params (fallback defaults)
LABEL_W_PRIMARY = getattr(CFG, "LABEL_WEIGHT_PRIMARY", 0.95)
LABEL_W_BENCH = getattr(CFG, "LABEL_WEIGHT_BENCH", 0.05)
RARE_WEIGHT = getattr(CFG, "RARE_WEIGHT", 2.0)
PSEUDO_WEIGHT = getattr(CFG, "PSEUDO_WEIGHT", 0.5)
RARE_COUNT_THRESHOLD = getattr(CFG, "RARE_COUNT_THRESHOLD", 20)

# ────────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────────────────────────

def _setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

# ────────────────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────────────────

def _md5(fp: Path) -> str:
    """
    Compute an MD5 hash of a file’s contents.
    """
    h = hashlib.md5()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _deduplicate(paths: Sequence[Path]) -> List[Path]:
    """
    Return unique file paths in first-seen order by file MD5.
    """
    seen_hashes: set[str] = set()
    unique: List[Path] = []
    for p in paths:
        try:
            sig = _md5(p)
        except Exception:
            # if hashing fails, just keep the path
            unique.append(p)
            continue
        if sig not in seen_hashes:
            seen_hashes.add(sig)
            unique.append(p)
    return unique


def _np_save(fp: Path, arr: np.ndarray) -> None:
    """Save a numpy array safely, creating parent dirs."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp, arr.astype(np.float32), allow_pickle=False)

# ────────────────────────────────────────────────────────────────────────────────
# Soft‑label construction
# ────────────────────────────────────────────────────────────────────────────────

def _secondary_list(raw: str | float | int) -> List[str]:
    if isinstance(raw, str) and raw:
        return [s for s in raw.split(";") if s]
    return []


def build_soft_label(
    primary: str,
    secondaries: List[str],
    bench_model: Optional[torch.nn.Module] = None,
    wav: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compose a normalized {species: prob} dict summing to 1,
    mixing primary/secondary ground truth with benchmark smoothing.
    """
    # base weights
    label: Dict[str, float] = defaultdict(float)
    rem = 1.0 - LABEL_W_BENCH
    if secondaries:
        sec_share = rem - LABEL_W_PRIMARY
        sec_w = sec_share / len(secondaries)
        for s in secondaries:
            label[s] += sec_w
        label[primary] += LABEL_W_PRIMARY
    else:
        label[primary] += rem

    # benchmark smoothing
    if bench_model is not None and wav is not None:
        mel = compute_mel(wav)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = bench_model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        for i, p in enumerate(probs):
            if p > 0:
                label[ALL_CLASSES[i]] += LABEL_W_BENCH * float(p)

    # normalize for safety
    total = sum(label.values())
    for k in list(label):
        label[k] /= total
    return dict(label)

# ────────────────────────────────────────────────────────────────────────────────
# Core processing functions
# ────────────────────────────────────────────────────────────────────────────────

def _process_recordings() -> None:
    """Process label‑verified *train_audio* recordings into 10‑s mel chunks."""
    log = logging.getLogger()
    log.info("✨ Processing labelled recordings …")

    # Load metadata CSV
    df_meta = pd.read_csv(CFG.TRAIN_CSV)

    # Quality gate
    if getattr(CFG, "MIN_RATING", 0) > 0 and "rating" in df_meta.columns:
        df_meta = df_meta[df_meta["rating"] >= CFG.MIN_RATING]

    # Map to existing files
    records: List[dict] = []
    for r in df_meta.itertuples(index=False):
        fp = CFG.TRAIN_AUDIO_DIR / r.filename
        if not fp.exists():
            log.warning("Missing %s", fp)
            continue
        d = r._asdict()  # type: ignore[attr-defined]
        d["filepath"] = fp
        records.append(d)
    df = pd.DataFrame.from_records(records)
    if df.empty:
        log.warning("No recordings found – aborting train stage")
        return

    # Deduplicate identical files
    df = df[df["filepath"].isin(_deduplicate(df["filepath"].tolist()))].reset_index(drop=True)

    # Compute noise score & select fold‑0 subset
    noise_scores: List[float] = []
    for fp in df["filepath"]:
        y = load_audio(fp)
        noise_scores.append(compute_noise_metric(y))
    df["noise_score"] = noise_scores

    ratio = getattr(CFG, "FOLD0_RATIO", 1.0)
    if 0 < ratio < 1.0:
        thresh = np.quantile(df["noise_score"], ratio)
        df = df[df["noise_score"] <= thresh].reset_index(drop=True)
        log.info("Selected %d clean recordings (%.0f%%)", len(df), 100 * ratio)

    # Load single-model benchmark for smoothing & swap logic
    bench_model = _load_benchmark()
    if bench_model:
        log.info("Benchmark model loaded for smoothing and swap logic.")
    vad_model, vad_ts = load_vad()

    # Prepare output directories
    mel_root = CFG.PROCESSED_DIR / "mels" / "train"
    lbl_root = CFG.PROCESSED_DIR / "labels" / "train"
    mel_root.mkdir(parents=True, exist_ok=True)
    lbl_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for rec in df.itertuples(index=False):
        y = load_audio(rec.filepath)
        y = trim_silence(y)
        y = remove_speech(y, vad_model, vad_ts)
        if np.sqrt((y ** 2).mean()) < CFG.RMS_THRESHOLD:
            continue  # too quiet

        secondaries = _secondary_list(getattr(rec, "secondary_labels", ""))
        primary_lbl = rec.primary_label

        # Swap logic using full-clip benchmark prediction
        if bench_model is not None:
            mel_full = compute_mel(y)
            x_full = torch.from_numpy(mel_full).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits_full = bench_model(x_full)
                probs_full = torch.sigmoid(logits_full)[0].cpu().numpy()
            top_lbl = ALL_CLASSES[int(probs_full.argmax())]
            if top_lbl != primary_lbl and top_lbl in secondaries:
                log.info("Swapping primary %s → %s", primary_lbl, top_lbl)
                primary_lbl = top_lbl
            elif top_lbl != primary_lbl and top_lbl not in secondaries:
                log.debug("Discarding mislabeled clip %s", rec.filepath.name)
                continue

        # Segment into fixed-length chunks
        for start_sec, chunk in segment_audio(y):
            if len(chunk) < CFG.TRAIN_CHUNK_SEC * CFG.SAMPLE_RATE:
                pad = CFG.TRAIN_CHUNK_SEC * CFG.SAMPLE_RATE - len(chunk)
                chunk = np.pad(chunk, (0, pad), mode="wrap")

            mel_chunk = compute_mel(chunk)
            soft = build_soft_label(primary_lbl, secondaries, bench_model, chunk)

            base = f"{rec.filepath.stem}_{int(start_sec)}s"
            mel_fp = mel_root / primary_lbl / f"{base}.npy"
            lbl_fp = lbl_root / primary_lbl / f"{base}.label.npy"

            _np_save(mel_fp, mel_chunk)

            vec = np.zeros(len(ALL_CLASSES), dtype=np.float32)
            for k, v in soft.items():
                vec[CLASS2IDX[k]] = v
            _np_save(lbl_fp, vec)

            rows.append({
                "mel_path": str(mel_fp.relative_to(CFG.PROCESSED_DIR)),
                "label_path": str(lbl_fp.relative_to(CFG.PROCESSED_DIR)),
                "label_json": json.dumps(soft, separators=(",",":")),
                "duration": CFG.TRAIN_CHUNK_SEC,
                "rating": getattr(rec, "rating", ""),
                "noise_score": rec.noise_score,  # type: ignore[attr-defined]
                "weight": 1.0,
            })

    out_csv = CFG.PROCESSED_DIR / "train_metadata.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("Saved %d mel chunks → %s", len(rows), out_csv.name)


# Pseudo‑labelling soundscapes ----------------------------------------------------

def _process_soundscapes() -> None:
    log = logging.getLogger()
    if not CFG.TRAIN_SOUNDSCAPE_DIR.exists():
        log.info("No soundscape directory – skipping pseudo‑labelling stage")
        return
    bench_model = _load_benchmark()
    if bench_model is None:
        log.warning("Benchmark unavailable – skipping soundscape pseudo‑labelling")
        return
    vad_model, vad_ts = load_vad()

    mel_root = CFG.PROCESSED_DIR / "mels" / "soundscape"
    lbl_root = CFG.PROCESSED_DIR / "labels" / "soundscape"
    mel_root.mkdir(parents=True, exist_ok=True)
    lbl_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for fp in sorted(CFG.TRAIN_SOUNDSCAPE_DIR.glob("*.ogg")):
        y = load_audio(fp)
        y = trim_silence(y)
        y = remove_speech(y, vad_model, vad_ts)

        for start_sec, seg in segment_audio(y):
            if len(seg) < CFG.TRAIN_CHUNK_SEC * CFG.SAMPLE_RATE:
                seg = np.pad(seg, (0, CFG.TRAIN_CHUNK_SEC * CFG.SAMPLE_RATE - len(seg)), mode="wrap")

            mel_seg = compute_mel(seg)
            x_seg = torch.from_numpy(mel_seg).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits_seg = bench_model(x_seg)
                probs = torch.sigmoid(logits_seg)[0].cpu().numpy()

            if float(probs.max()) < CFG.PSEUDO_THRESHOLD:
                continue

            mel_fp = mel_root / f"{fp.stem}_{int(start_sec)}s.npy"
            lbl_fp = lbl_root / f"{fp.stem}_{int(start_sec)}s.label.npy"
            _np_save(mel_fp, mel_seg)
            _np_save(lbl_fp, probs.astype(np.float32))

            soft = {ALL_CLASSES[i]: float(p) for i, p in enumerate(probs) if p > 0}
            rows.append({
                "mel_path": str(mel_fp.relative_to(CFG.PROCESSED_DIR)),
                "label_path": str(lbl_fp.relative_to(CFG.PROCESSED_DIR)),
                "label_json": json.dumps(soft, separators=(",",":")),
                "duration": CFG.TRAIN_CHUNK_SEC,
                "rating": "pseudo",
                "noise_score": compute_noise_metric(seg),
                "weight": PSEUDO_WEIGHT,
            })

    if rows:
        out_csv = CFG.PROCESSED_DIR / "soundscape_metadata.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("Saved %d pseudo‑labelled chunks → %s", len(rows), out_csv.name)


def _apply_rare_weighting() -> None:
    meta_files = [
        CFG.PROCESSED_DIR / "train_metadata.csv",
        CFG.PROCESSED_DIR / "soundscape_metadata.csv",
    ]
    dfs = [pd.read_csv(p) for p in meta_files if p.exists()]
    if not dfs:
        return

    all_meta = pd.concat(dfs, ignore_index=True)
    counts = Counter(max(json.loads(js), key=json.loads(js).get) for js in all_meta["label_json"])
    rare_species = {sp for sp, c in counts.items() if c < RARE_COUNT_THRESHOLD}

    for p in meta_files:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["weight"] = [
            RARE_WEIGHT if max(json.loads(js), key=json.loads(js).get) in rare_species else 1.0
            for js in df["label_json"]
        ]
        df.to_csv(p, index=False)

    logging.getLogger().info(
        "Applied rare‑species weighting (x%.1f) to %d species", RARE_WEIGHT, len(rare_species)
    )


def main() -> None:
    pa = argparse.ArgumentParser(description="Preprocess BirdCLEF 2025 audio")
    pa.add_argument("--verbose", action="store_true", help="debug logging")
    args = pa.parse_args()

    _setup_logger(args.verbose)
    seed_everything(getattr(CFG, "SEED", 42))

    CFG.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    _process_recordings()
    _process_soundscapes()
    _apply_rare_weighting()

    logging.info("✅ Preprocessing finished – data saved to %s", CFG.PROCESSED_DIR)

if __name__ == "__main__":
    main()

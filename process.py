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
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

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

# ── Benchmark *fold‑ensemble* helper ─────────────────────────────────────────
class BenchEnsemble:
    """
    Averages `softmax` probabilities from every `model_fold*.pth` found in a
    directory.  Each checkpoint must load to a *complete* `nn.Module` that
    returns raw **logits** for a mono waveform (shape [1,T]).
    """
    def __init__(self, ckpt_dir: Path):
        self.nets = []
        self.classes = None
        for p in ckpt_dir.glob("model_fold*.pth"):
            m = torch.load(p, map_location="cpu")
            # handle both full objects and dict bundles
            if isinstance(m, dict) and "model" in m:
                m = m["model"]
            m.eval()
            self.nets.append(m)
            if self.classes is None:
                if hasattr(m, "classes"):
                    self.classes = list(m.classes)
                elif isinstance(m, dict) and "species2idx" in m:
                    self.classes = list(m["species2idx"].keys())
        if not self.nets:
            raise FileNotFoundError(f"No model_fold*.pth in {ckpt_dir}")
        if self.classes is None:
            raise RuntimeError("Couldn’t infer class list from checkpoints")

    # ── inside class BenchEnsemble ─────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, wav: np.ndarray) -> np.ndarray:
        """
        Return averaged probability vector (length = n_classes) across all folds.
    
        Parameters
        ----------
        wav : np.ndarray
            1‑D mono waveform @ 32 kHz (float32).  The helper turns it into the
            128 × T mel that the checkpoints expect.
        """
        # 1) convert → mel (shape [1, 1, 128, T]) on CPU
        mel = compute_mel(wav)                                    # (128, T)
        mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)     # (1,1,128,T)
    
        # 2) forward through every fold → logits
        logits = [net(mel) for net in self.nets]                  # list[1, 206]
    
        # 3) softmax → probs  and   mean across folds
        probs = [torch.softmax(l, dim=1).cpu().numpy()[0] for l in logits]  # list[206]
        return np.mean(probs, axis=0)                             # (206,)


def _discover_classes() -> List[str]:
    """Return the full list of 206 BirdCLEF classes.

    Priority order:
      1. ``CFG.CLASSES`` – user‑supplied explicit ordering (recommended)
      2. taxonomy CSV (if header contains "primary_label")
      3. ``train.csv`` primary_label column
    """
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
RARE_WEIGHT = getattr(CFG, "RARE_WEIGHT", 2.0)  # multiplied in weighting step
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


def _np_save(fp: Path, arr: np.ndarray) -> None:
    """Safe ``np.save`` that creates parent directories."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp, arr.astype(np.float32), allow_pickle=False)

def _load_benchmark():
    """
    Returns either:
      • BenchEnsemble(dir)  if CFG.BENCHMARK_MODEL points to a directory, or
      • torch.load(file)    for a single‑file benchmark.
    """
    if not CFG.BENCHMARK_MODEL:
        return None
    bp = Path(CFG.BENCHMARK_MODEL)
    if not bp.exists():
        logging.warning("Benchmark path %s missing – skipping.", bp)
        return None
    return BenchEnsemble(bp) if bp.is_dir() else torch.load(bp, map_location="cpu")


# Deduplication based on MD5 ------------------------------------------------------


def _md5(fp: Path) -> str:
    h = hashlib.md5()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _deduplicate(paths: Sequence[Path]) -> List[Path]:
    """Return *unique* filepaths preserving first appearance order."""
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in paths:
        try:
            sig = _md5(p)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Hash failed for %s – keeping (%s)", p.name, exc)
            uniq.append(p)
            continue
        if sig not in seen:
            seen.add(sig)
            uniq.append(p)
    return uniq


# Soft‑label construction ---------------------------------------------------------


def _secondary_list(raw: str | float | int) -> List[str]:
    if isinstance(raw, str) and raw:
        return [s for s in raw.split(";") if s]
    return []


def build_soft_label(
    primary: str,
    secondaries: List[str],
    bench_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compose a normalised ``{species: prob}`` dict that sums to 1."""
    label: Dict[str, float] = defaultdict(float)

    # ── primary / secondary share ─────────────────────────────────────────────
    rem = 1.0 - LABEL_W_BENCH
    if secondaries:
        sec_share = rem - LABEL_W_PRIMARY
        sec_w = sec_share / len(secondaries)
        for s in secondaries:
            label[s] += sec_w
        label[primary] += LABEL_W_PRIMARY
    else:
        label[primary] += rem

    # ── benchmark smoothing ──────────────────────────────────────────────────
    if bench_probs is not None and len(bench_probs) == len(ALL_CLASSES):
        for idx, p in enumerate(bench_probs):
            if p > 0:
                label[ALL_CLASSES[idx]] += LABEL_W_BENCH * float(p)

    # normalise (safety)
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

    # Load metadata CSV -------------------------------------------------------
    df_meta = pd.read_csv(CFG.TRAIN_CSV)

    # Quality gate
    if getattr(CFG, "MIN_RATING", 0) > 0 and "rating" in df_meta.columns:
        df_meta = df_meta[df_meta["rating"] >= CFG.MIN_RATING]

    # Map to existing files
    records: List[dict] = []
    for r in df_meta.itertuples(index=False):
        fp = CFG.TRAIN_AUDIO_DIR / r.primary_label / r.filename
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

    bench_model = _load_benchmark()
    if bench_model:
        log.info("Benchmark loaded (%s folds)", getattr(bench_model, "nets", [bench_model]).__len__())


    vad_model, vad_ts = load_vad()

    # Output dirs
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
        bench_probs: Optional[np.ndarray] = None

        if bench_model is not None:
            with torch.no_grad():
                bench_probs = bench_model.predict_proba(y)  # type: ignore[attr-defined]
            top_lbl = bench_model.classes[int(bench_probs.argmax())]  # type: ignore[attr-defined]
            # swap primary ↔ secondary if benchmark strongly disagrees
            if top_lbl != primary_lbl and top_lbl in secondaries:
                log.info("Swapping primary %s → %s", primary_lbl, top_lbl)
                primary_lbl = top_lbl
            elif top_lbl != primary_lbl and top_lbl not in secondaries:
                # Likely mis‑labelled, skip clip entirely
                log.debug("Discarding mislabeled clip %s", rec.filepath.name)
                continue

        for start_sec, chunk in segment_audio(y):
            if len(chunk) < CFG.TRAIN_CHUNK_SEC * CFG.SAMPLE_RATE:
                pad = CFG.TRAIN_CHUNK_SEC * CFG.SAMPLE_RATE - len(chunk)
                chunk = np.pad(chunk, (0, pad), mode="wrap")

            mel = compute_mel(chunk)
            soft = build_soft_label(primary_lbl, secondaries, bench_probs)

            # save artefacts ---------------------------------------------------
            base = f"{rec.filepath.stem}_{int(start_sec)}s"
            mel_fp = mel_root / primary_lbl / f"{base}.npy"
            lbl_fp = lbl_root / primary_lbl / f"{base}.label.npy"

            _np_save(mel_fp, mel)

            vec = np.zeros(len(ALL_CLASSES), dtype=np.float32)
            for k, v in soft.items():
                vec[CLASS2IDX[k]] = v
            _np_save(lbl_fp, vec)

            rows.append(
                {
                    "mel_path": str(mel_fp.relative_to(CFG.PROCESSED_DIR)),
                    "label_path": str(lbl_fp.relative_to(CFG.PROCESSED_DIR)),
                    "label_json": json.dumps(soft, separators=(",", ":")),
                    "duration": CFG.TRAIN_CHUNK_SEC,
                    "rating": getattr(rec, "rating", ""),
                    "noise_score": rec.noise_score,  # type: ignore[attr-defined]
                    "weight": 1.0,
                }
            )

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

            with torch.no_grad():
                probs = bench_model.predict_proba(seg)  # type: ignore[attr-defined]
            if float(probs.max()) < CFG.PSEUDO_THRESHOLD:
                continue

            mel = compute_mel(seg)
            base = f"{fp.stem}_{int(start_sec)}s"
            mel_fp = mel_root / f"{base}.npy"
            lbl_fp = lbl_root / f"{base}.label.npy"
            _np_save(mel_fp, mel)
            _np_save(lbl_fp, probs.astype(np.float32))

            soft = {ALL_CLASSES[i]: float(p) for i, p in enumerate(probs) if p > 0}
            rows.append(
                {
                    "mel_path": str(mel_fp.relative_to(CFG.PROCESSED_DIR)),
                    "label_path": str(lbl_fp.relative_to(CFG.PROCESSED_DIR)),
                    "label_json": json.dumps(soft, separators=(",", ":")),
                    "duration": CFG.TRAIN_CHUNK_SEC,
                    "rating": "pseudo",
                    "noise_score": compute_noise_metric(seg),
                    "weight": PSEUDO_WEIGHT,
                }
            )

    if rows:
        out_csv = CFG.PROCESSED_DIR / "soundscape_metadata.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("Saved %d pseudo‑labelled chunks → %s", len(rows), out_csv.name)


# Rare‑species sample weighting ---------------------------------------------------


def _apply_rare_weighting() -> None:
    meta_files = [
        CFG.PROCESSED_DIR / "train_metadata.csv",
        CFG.PROCESSED_DIR / "soundscape_metadata.csv",
    ]
    dfs: List[pd.DataFrame] = [pd.read_csv(p) for p in meta_files if p.exists()]
    if not dfs:
        return

    all_meta = pd.concat(dfs, ignore_index=True)
    counts = Counter(max(json.loads(js), key=json.loads(js).get) for js in all_meta["label_json"])
    rare_species = {sp for sp, c in counts.items() if c < RARE_COUNT_THRESHOLD}

    for p in meta_files:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["weight"] = [RARE_WEIGHT if max(json.loads(js), key=json.loads(js).get) in rare_species else 1.0 for js in df["label_json"]]
        df.to_csv(p, index=False)

    logging.getLogger().info("Applied rare‑species weighting (x%.1f) to %d species", RARE_WEIGHT, len(rare_species))


# ────────────────────────────────────────────────────────────────────────────────
# Entry‑point
# ────────────────────────────────────────────────────────────────────────────────


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

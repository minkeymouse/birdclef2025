"""Full pseudo label update pipeline (8h budget) — iter 1 + iter 2.

Phase 1: train exp185 (NS iter 1) folds 0-4 (skip if ckpt exists)
Phase 2: local validation (read training history)
Phase 3: generate iter 2 pseudo from exp185 5-ckpt teacher + power transform
Phase 4: train exp186 (NS iter 2) fold 0 smoke
Phase 5: iter 2 local val
"""

from __future__ import annotations
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
import subprocess

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.sed.config import exp185_noisy_student_full, SedConfig, PseudoConfig
from experiments.sed.train import train_one_fold

OUT_EXP185 = ROOT / "experiments/_data_pipelines/exp185_outputs"
OUT_EXP186 = ROOT / "experiments/_data_pipelines/exp186_outputs"
STATUS_FILE = ROOT / "experiments/sed/pseudo_pipeline_status.json"
PSEUDO_ITER2_NPZ = ROOT / "experiments/sed/pseudo_noisy_student_iter2_p0.70.npz"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def update_status(phase: str, status: str, **extra):
    state = {"updated": datetime.now().isoformat(), "phase": phase, "status": status, **extra}
    if STATUS_FILE.exists():
        try:
            prev = json.loads(STATUS_FILE.read_text())
            phase_key = f"phase_{phase}_history"
            prev.setdefault(phase_key, []).append(state)
            prev.update(state)
            state = prev
        except: pass
    STATUS_FILE.write_text(json.dumps(state, indent=2))


def phase1_train_iter1():
    log("=== PHASE 1: train exp185 folds 0-4 ===")
    cfg = exp185_noisy_student_full()
    t0 = time.time()
    for f in range(5):
        ckpt = OUT_EXP185 / f"fold{f}" / "best_ckpt.pt"
        if ckpt.exists():
            log(f"fold {f} ckpt exists — skipping")
            continue
        log(f"--- training fold {f} (elapsed {time.time()-t0:.0f}s) ---")
        try:
            train_one_fold(cfg, fold_id=f)
            update_status("phase1", "ok", last_fold=f)
        except Exception:
            traceback.print_exc()
            update_status("phase1", "failed", failed_fold=f)
            return False
    update_status("phase1", "complete", elapsed_min=(time.time()-t0)/60)
    log(f"Phase 1 done in {(time.time()-t0)/60:.1f} min")
    return True


def phase2_iter1_validation():
    log("=== PHASE 2: iter 1 local validation ===")
    val_ss_per_fold = {}
    val_ta_per_fold = {}
    for f in range(5):
        hist_path = OUT_EXP185 / f"fold{f}" / "history.json"
        if not hist_path.exists():
            log(f"fold {f} history missing")
            continue
        h = json.loads(hist_path.read_text())
        hist = h["history"] if isinstance(h, dict) and "history" in h else h
        best = max(hist, key=lambda r: r.get("val_SS", -1))
        val_ss_per_fold[f] = best.get("val_SS")
        val_ta_per_fold[f] = best.get("val_TA")
        log(f"fold {f}: val_SS={best.get('val_SS'):.4f} val_TA={best.get('val_TA'):.4f}")

    if val_ss_per_fold:
        mean_ss = sum(val_ss_per_fold.values()) / len(val_ss_per_fold)
        mean_ta = sum(val_ta_per_fold.values()) / len(val_ta_per_fold)
        log(f"mean val_SS: {mean_ss:.4f}, mean val_TA: {mean_ta:.4f}")
        update_status("phase2", "complete",
                      mean_val_ss=mean_ss, mean_val_ta=mean_ta,
                      val_ss_per_fold=val_ss_per_fold)
    return True


def phase3_gen_iter2_pseudo():
    log("=== PHASE 3: generate iter 2 pseudo from exp185 5-ckpt teacher ===")
    if PSEUDO_ITER2_NPZ.exists():
        log(f"iter 2 pseudo already exists at {PSEUDO_ITER2_NPZ} — skipping")
        return True

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from collections import defaultdict
    from experiments.sed.model import DistilledSED
    from experiments.sed.data import load_audio, DATA

    SR = 32_000; CHUNK_S = 5; FILE_DUR_S = 60
    N_CHUNKS = FILE_DUR_S // CHUNK_S
    N_CLS = 234
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load exp185 5 ckpts as teacher
    cfg = exp185_noisy_student_full()
    models = []
    for f in range(5):
        ck = OUT_EXP185 / f"fold{f}" / "best_ckpt.pt"
        if not ck.exists():
            log(f"missing teacher ckpt fold {f}")
            return False
        m = DistilledSED(cfg, n_cls=N_CLS).to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        elif "state_dict" in sd: sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        m.eval()
        models.append(m)
    log(f"loaded {len(models)} teacher ckpts")

    # Load all unlabeled SS files, EXCLUDE labeled ones
    SS_DIR = DATA / "train_soundscapes"
    all_files = sorted([f.name for f in SS_DIR.glob("*.ogg")])
    lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labeled_files = set(lbl["filename"].unique())
    unlabeled_files = [f for f in all_files if f not in labeled_files]
    log(f"unlabeled SS files: {len(unlabeled_files)} (total {len(all_files)}, labeled {len(labeled_files)})")

    class SSFileDataset(Dataset):
        def __init__(self, files): self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            wav = load_audio(SS_DIR / self.files[idx], SR)
            target = FILE_DUR_S * SR
            if wav is None: wav = np.zeros(target, dtype=np.float32)
            elif len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))
            else: wav = wav[:target]
            chunks = wav.reshape(N_CHUNKS, CHUNK_S * SR).astype(np.float32)
            return idx, torch.from_numpy(chunks)

    ds = SSFileDataset(unlabeled_files)
    loader = DataLoader(ds, batch_size=4, num_workers=4, pin_memory=True, shuffle=False)

    # Run ensemble inference (match generate_pseudo_v12_balanced.py pattern)
    all_probs = []
    file_idx = []
    end_secs = []
    t0 = time.time()
    n_done = 0
    with torch.inference_mode():
        for batch_file_idx, batch_wav in loader:
            B, C, S = batch_wav.shape
            wav = batch_wav.to(DEVICE).reshape(B * C, S)
            probs_sum = torch.zeros(B * C, N_CLS, device=DEVICE)
            for m in models:
                out = m(wav)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                probs_sum += torch.sigmoid(logits)
            probs_avg = (probs_sum / len(models)).cpu().numpy()
            for bi in range(B):
                pi = probs_avg[bi*C:(bi+1)*C]
                all_probs.append(pi)
                for ci in range(C):
                    file_idx.append(int(batch_file_idx[bi]))
                    end_secs.append((ci + 1) * CHUNK_S)
            n_done += B
            if n_done % 200 == 0:
                log(f"inference {n_done}/{len(unlabeled_files)} ({time.time()-t0:.0f}s)")
    log(f"inference done in {(time.time()-t0)/60:.1f} min")

    all_probs = np.concatenate(all_probs, axis=0)
    file_idx_arr = np.array(file_idx, dtype=np.int32)
    end_secs_arr = np.array(end_secs, dtype=np.int32)
    log(f"total chunks: {len(all_probs)}, shape: {all_probs.shape}")

    # Selection: max prob >= 0.5, classes < 0.1 zero
    max_p = all_probs.max(axis=1)
    keep = max_p >= 0.5
    kept_probs = np.where(all_probs < 0.1, 0.0, all_probs)[keep]
    kept_file_idx = file_idx_arr[keep]
    kept_end_secs = end_secs_arr[keep]
    log(f"after selection (max>=0.5, zero<0.1): {len(kept_probs)} chunks")

    # Per-class cap = 1500
    PER_CLASS_CAP = 1500
    rng = np.random.default_rng(42)
    primary_cls = kept_probs.argmax(axis=1)
    keep_mask = np.zeros(len(kept_probs), dtype=bool)
    for cls in range(N_CLS):
        idx_cls = np.where(primary_cls == cls)[0]
        if len(idx_cls) <= PER_CLASS_CAP:
            keep_mask[idx_cls] = True
        else:
            sel = rng.choice(idx_cls, size=PER_CLASS_CAP, replace=False)
            keep_mask[sel] = True
    kept_probs = kept_probs[keep_mask]
    kept_file_idx = kept_file_idx[keep_mask]
    kept_end_secs = kept_end_secs[keep_mask]
    log(f"after per-class cap {PER_CLASS_CAP}: {len(kept_probs)} chunks")

    # Power transform alpha=0.7
    POWER = 0.7
    kept_probs_ns = np.clip(kept_probs, 1e-6, 1 - 1e-6) ** POWER
    confidence_weights = kept_probs_ns.sum(axis=1).astype(np.float32)
    log(f"power transform applied (p ** {POWER})")
    log(f"  confidence_weights mean={confidence_weights.mean():.3f}")

    # Save
    files_arr = np.array(unlabeled_files, dtype=object)
    np.savez(PSEUDO_ITER2_NPZ,
             files=files_arr, kept_file_idx=kept_file_idx,
             kept_end_secs=kept_end_secs, kept_probs=kept_probs_ns.astype(np.float32),
             confidence_weights=confidence_weights,
             power_alpha=np.array([POWER], dtype=np.float32))
    log(f"saved {PSEUDO_ITER2_NPZ}")
    update_status("phase3", "complete", n_chunks=int(len(kept_probs_ns)))
    return True


def exp186_iter2_config():
    """Same as exp185 but with iter2 pseudo."""
    from dataclasses import replace
    cfg = exp185_noisy_student_full()
    cfg = replace(cfg,
                  name="exp186_noisy_student_iter2",
                  output_dir=str(OUT_EXP186),
                  pseudo=PseudoConfig(
                      pseudo_npz=str(PSEUDO_ITER2_NPZ.relative_to(ROOT)),
                      pseudo_share_per_batch=0.40,
                  ),
                  notes="exp186 = NS iter 2 with exp185 5-ckpt teacher")
    return cfg


def phase4_train_iter2_smoke():
    log("=== PHASE 4: train exp186 (NS iter 2) fold 0 smoke ===")
    cfg = exp186_iter2_config()
    t0 = time.time()
    try:
        train_one_fold(cfg, fold_id=0)
        update_status("phase4", "complete", elapsed_min=(time.time()-t0)/60)
        log(f"Phase 4 done in {(time.time()-t0)/60:.1f} min")
    except Exception:
        traceback.print_exc()
        update_status("phase4", "failed")
        return False
    return True


def phase5_iter2_validation():
    log("=== PHASE 5: iter 2 local validation ===")
    hist_path = OUT_EXP186 / "fold0" / "history.json"
    if not hist_path.exists():
        log("iter 2 history missing")
        return False
    h = json.loads(hist_path.read_text())
    hist = h["history"] if isinstance(h, dict) and "history" in h else h
    best = max(hist, key=lambda r: r.get("val_SS", -1))
    log(f"iter 2 fold 0: val_SS={best.get('val_SS'):.4f} val_TA={best.get('val_TA'):.4f}")

    # compare to iter 1 fold 0
    iter1_h = OUT_EXP185 / "fold0" / "history.json"
    if iter1_h.exists():
        h1 = json.loads(iter1_h.read_text())
        hist1 = h1["history"] if isinstance(h1, dict) and "history" in h1 else h1
        b1 = max(hist1, key=lambda r: r.get("val_SS", -1))
        log(f"iter 1 fold 0: val_SS={b1.get('val_SS'):.4f} val_TA={b1.get('val_TA'):.4f}")
        delta_ss = best["val_SS"] - b1["val_SS"]
        delta_ta = best["val_TA"] - b1["val_TA"]
        log(f"delta iter2 - iter1: val_SS={delta_ss:+.4f} val_TA={delta_ta:+.4f}")
        update_status("phase5", "complete",
                      iter2_val_ss=best.get("val_SS"), iter1_val_ss=b1.get("val_SS"),
                      delta_val_ss=delta_ss)
    return True


def main():
    OUT_EXP185.mkdir(parents=True, exist_ok=True)
    OUT_EXP186.mkdir(parents=True, exist_ok=True)
    update_status("init", "starting")

    t0 = time.time()
    log("=" * 60)
    log("Pseudo label update pipeline — 8h budget — iter 1 + iter 2")
    log("=" * 60)

    if not phase1_train_iter1(): return
    phase2_iter1_validation()
    log(f"\nElapsed so far: {(time.time()-t0)/60:.1f} min")

    if not phase3_gen_iter2_pseudo(): return
    log(f"\nElapsed so far: {(time.time()-t0)/60:.1f} min")

    if not phase4_train_iter2_smoke(): return
    phase5_iter2_validation()

    log(f"\nPipeline complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

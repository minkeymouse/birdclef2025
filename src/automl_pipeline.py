#!/usr/bin/env python3
"""
automl_pipeline.py – BirdCLEF‑2025 end‑to‑end AutoML orchestrator
=================================================================
This single entry‑point reproduces *exactly* the workflow we have
agreed on in chat – from high‑confidence “golden” chunk extraction
all the way to the final submission CSV.

The pipeline is **resumable**.  Each stage checks for its expected
artefacts before running, so you can safely interrupt / restart or
iterate more loops later.

CLI overview
------------
```bash
# 1) full initial run (golden → initial 6 models → soundscape pseudo‑labels)
python -m src.automl_pipeline --initial

# 2) add 3 refinement loops on top of existing state
python -m src.automl_pipeline --iterate 3
```

Implementation notes
-------------------
* Heavy work is off‑loaded to dedicated scripts already in *src/process*
  and *src/train* – this wrapper only orchestrates them.
* All subprocesses inherit *stdout* so you can tail the log in **tmux**.
* YAML config paths are centralised in the constants section.  If you
  move the config files, just update those constants.
* We convert checkpoints to **TorchScript** once per training script so
  the VAD‑less *process_pseudo.py* can `torch.jit.load()` them quickly.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

import yaml

# -----------------------------------------------------------------------------
# Constants – adjust if you move the YAML files
# -----------------------------------------------------------------------------
CFG_PROCESS = Path("config/process.yaml")
CFG_INIT_TRAIN = Path("config/initial_train.yaml")
CFG_TRAIN = Path("config/train.yaml")  # generic for later loops
CFG_INFER = Path("config/inference.yaml")

# Training entry points (within src/train)
TRAIN_EFF_SCRIPT = Path("src/train/train_efficientnet.py")
TRAIN_REG_SCRIPT = Path("src/train/train_regnety.py")

# Process scripts
PROC_GOLD = Path("src/process/process_gold.py")
PROC_RARE = Path("src/process/process_rare.py")
PROC_PSEUDO = Path("src/process/process_pseudo.py")

# Top‑level directories (read from process.yaml once)
with CFG_PROCESS.open() as f:
    _PROC_CFG = yaml.safe_load(f)
_DATA_ROOT = Path(_PROC_CFG["paths"]["data_root"]).expanduser()
PROCESSED_DIR = Path(_PROC_CFG["paths"]["processed_dir"]).expanduser()
MODELS_DIR = Path(_PROC_CFG["paths"]["processed_dir"]).parent / "models"
SOUNDSCAPE_DIR = _DATA_ROOT / "train_soundscapes"

# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
log = logging.getLogger("automl")


# -----------------------------------------------------------------------------
# Sub‑process wrapper
# -----------------------------------------------------------------------------

def _run(cmd: list[str] | str, check: bool = True):
    """Run *cmd* (list or str) and stream output. Abort on non‑zero exit."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    log.info("RUN → %s", display)
    start = time.time()
    try:
        subprocess.run(cmd, shell=isinstance(cmd, str), check=check)
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s", e)
        sys.exit(1)
    log.info("✔ finished in %.1fs", time.time() - start)


# -----------------------------------------------------------------------------
# Stage helpers
# -----------------------------------------------------------------------------

def stage_process_golden():
    """Generate the high‑confidence golden dataset (idempotent)."""
    meta_csv = Path(_PROC_CFG["paths"]["train_metadata"])
    if meta_csv.exists():
        log.info("train_metadata.csv already exists – skip golden extraction.")
        return
    _run(["python", str(PROC_GOLD)])
    # Rare stage is optional for *initial* but fast – include it here so the
    # first model already sees minorities.
    _run(["python", str(PROC_RARE)])


def stage_process_pseudo():
    """Run pseudo‑label generation on unseen recordings + update metadata."""
    _run(["python", str(PROC_PSEUDO)])


def stage_train(cfg_path: Path):
    """Train both ensembles as per *cfg_path*."""
    _run(["python", str(TRAIN_EFF_SCRIPT), "--cfg", str(cfg_path)])
    _run(["python", str(TRAIN_REG_SCRIPT), "--cfg", str(cfg_path)])

    # After training convert the *top‑3* checkpoints of each architecture to
    # TorchScript so *process_pseudo.py* can load them without class code.
    for arch in ("efficientnet_b0", "regnety_800mf"):
        arch_dir = MODELS_DIR / arch
        for pth in arch_dir.glob("*.pth"):
            ts_path = pth.with_suffix(".ts.pt")
            if ts_path.exists():
                continue
            _run(
                [
                    "python",
                    "- <<PY",
                    dedent(
                        f"""
                        import torch, sys, pathlib
                        p = pathlib.Path('{pth}');
                        ckpt = torch.load(p, map_location='cpu')
                        from torchvision import models
                        if '{arch}'.startswith('efficientnet'):
                            m = models.efficientnet_b0(weights=None)
                            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, ckpt['model_state_dict']['classifier.1.weight'].shape[0])
                        else:
                            m = models.regnet_y_800mf(weights=None)
                            m.fc = torch.nn.Linear(m.fc.in_features, ckpt['model_state_dict']['fc.weight'].shape[0])
                        m.load_state_dict(ckpt['model_state_dict'], strict=True)
                        m.eval()
                        torch.jit.script(m).save(str(p.with_suffix('.ts.pt')))
                        """,
                    ),
                    "PY",
                ],
                check=True,
            )


def stage_inference():
    """Run final inference on the public test soundscapes."""
    _run(["python", "src/inference.py"])


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End‑to‑end AutoML pipeline")
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Run golden extraction, initial 6‑model training and soundscape pseudo‑labeling.",
    )
    parser.add_argument(
        "--iterate",
        type=int,
        default=0,
        metavar="N",
        help="Run N additional refinement loops (rare + pseudo → re‑train).",
    )
    parser.add_argument(
        "--final-infer",
        action="store_true",
        help="After loops, run inference on the *test* set to create submission.csv.",
    )
    args = parser.parse_args()

    # -------------------- Stage 0: golden / initial ----------------------
    if args.initial:
        log.info("==== Stage 0 – golden extraction ====")
        stage_process_golden()

        log.info("==== Stage 1 – initial training (6 models) ====")
        stage_train(CFG_INIT_TRAIN)

        log.info("==== Stage 2 – first pseudo‑labels for soundscapes ====")
        stage_process_pseudo()

    # -------------------- Iterative refinement ---------------------------
    loops = args.iterate
    for i in range(1, loops + 1):
        log.info("==== ITERATION %d/%d – rare + pseudo + retrain ====", i, loops)
        stage_process_pseudo()  # adds new confident chunks to metadata

        # For later loops we switch to the generic train.yaml which now includes
        # *include_pseudo* / *include_synthetic* true by default.
        with CFG_TRAIN.open() as f:
            cfg = yaml.safe_load(f)
        cfg["dataset"]["include_pseudo"] = True
        cfg["dataset"]["include_synthetic"] = True
        tmp_cfg = PROCESSED_DIR / f"auto_train_loop{i}.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg))

        stage_train(tmp_cfg)

    # -------------------- Final public test inference -------------------
    if args.final_infer or (not args.initial and loops == 0):
        log.info("==== Final inference on public test soundscapes ====")
        stage_inference()

    log.info("🎉 Pipeline finished.")


if __name__ == "__main__":
    main()

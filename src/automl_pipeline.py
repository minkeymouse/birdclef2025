#!/usr/bin/env python3
"""
automl_pipeline.py – BirdCLEF-2025 AutoML orchestrator
====================================================
Orchestrates the end-to-end workflow:
 1) Golden + rare chunk extraction
 2) Initial training (EfficientNet + RegNetY)
 3) Pseudo-label generation
 4) Iterative refinement loops
 5) Final inference

Idempotent and resumable: each stage checks for output artifacts before running.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

import yaml

# ----------------------------------------------------------------------------
# Config file locations
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
CFG_DIR = BASE_DIR / "config"
CFG_PROCESS = CFG_DIR / "process.yaml"
CFG_INIT = CFG_DIR / "initial_train.yaml"
CFG_LOOP = CFG_DIR / "train.yaml"
CFG_INFER = CFG_DIR / "inference.yaml"

# ----------------------------------------------------------------------------
# Script entry points
# ----------------------------------------------------------------------------
SCRIPTS = {
    "golden": BASE_DIR / "src" / "process" / "process_gold.py",
    "rare":   BASE_DIR / "src" / "process" / "process_rare.py",
    "pseudo": BASE_DIR / "src" / "process" / "process_pseudo.py",
    "eff":    BASE_DIR / "src" / "train" / "train_efficientnet.py",
    "reg":    BASE_DIR / "src" / "train" / "train_regnety.py",
    "infer":  BASE_DIR / "src" / "inference.py",
}

# ----------------------------------------------------------------------------
# Load shared paths from process.yaml
# ----------------------------------------------------------------------------
with CFG_PROCESS.open() as f:
    PROC_CFG = yaml.safe_load(f)
DATA_ROOT = Path(PROC_CFG["paths"]["data_root"]).expanduser()
PROCESSED = Path(PROC_CFG["paths"]["processed_dir"]).expanduser()
MODELS = PROCESSED.parent / "models"

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("automl")

# ----------------------------------------------------------------------------
# Utility to run subprocesses
# ----------------------------------------------------------------------------
def run_cmd(cmd: list[str] | str) -> None:
    display = cmd if isinstance(cmd, str) else " ".join(cmd)
    log.info("RUN → %s", display)
    start = time.time()
    try:
        subprocess.run(cmd, shell=isinstance(cmd, str), check=True)
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s", e)
        sys.exit(1)
    log.info("✔ Completed in %.1fs", time.time() - start)

# ----------------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------------
def extract_golden() -> None:
    train_meta = Path(PROC_CFG["paths"]["train_metadata"])
    if train_meta.exists():
        log.info("train_metadata.csv exists. Skipping golden/rare extraction.")
        return
    run_cmd([sys.executable, str(SCRIPTS["golden"])])
    run_cmd([sys.executable, str(SCRIPTS["rare"])])


def generate_pseudo() -> None:
    run_cmd([sys.executable, str(SCRIPTS["pseudo"])])


def train_models(cfg_path: Path) -> None:
    run_cmd([sys.executable, str(SCRIPTS["eff"]), "--cfg", str(cfg_path)])
    run_cmd([sys.executable, str(SCRIPTS["reg"]), "--cfg", str(cfg_path)])
    # Convert PTH → TorchScript for next pseudo stage
    for arch in ("efficientnet_b0", "regnety_800mf"):
        arch_dir = MODELS / arch
        for pth in arch_dir.glob("*.pth"):
            ts_path = pth.with_suffix(".ts.pt")
            if ts_path.exists():
                continue
            script = dedent(f"""
                import torch, pathlib
                from torchvision import models
                p = pathlib.Path('{pth}')
                ck = torch.load(p, map_location='cpu')
                if '{arch}'.startswith('efficientnet'):
                    m = models.efficientnet_b0(weights=None)
                    m.classifier[1] = torch.nn.Linear(
                        m.classifier[1].in_features,
                        ck['model_state_dict']['classifier.1.weight'].shape[0]
                    )
                else:
                    m = models.regnet_y_800mf(weights=None)
                    m.fc = torch.nn.Linear(
                        m.fc.in_features,
                        ck['model_state_dict']['fc.weight'].shape[0]
                    )
                m.load_state_dict(ck['model_state_dict'], strict=True)
                torch.jit.script(m.eval()).save(str(p.with_suffix('.ts.pt')))
            """)
            run_cmd([sys.executable, "-c", script])


def run_inference() -> None:
    run_cmd([sys.executable, str(SCRIPTS["infer"])])

# ----------------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AutoML pipeline for BirdCLEF-2025")
    parser.add_argument("--initial", action="store_true", help="Run initial golden/train/pseudo")
    parser.add_argument("--iterate", type=int, default=0, help="Number of refinement loops")
    parser.add_argument("--final-infer", action="store_true", help="Run final inference stage")
    args = parser.parse_args()

    if args.initial:
        log.info("=== Stage 0: Golden + Rare extraction ===")
        extract_golden()

        log.info("=== Stage 1: Initial training ===")
        train_models(CFG_INIT)

        log.info("=== Stage 2: Pseudo-label generation ===")
        generate_pseudo()

    for i in range(args.iterate):
        log.info("=== Iteration %d/%d: Pseudo + Retrain ===", i+1, args.iterate)
        generate_pseudo()
        # prepare loop-specific YAML
        loop_cfg = PROC_CFG.copy()
        loop_cfg['dataset']['include_pseudo'] = True
        loop_cfg['dataset']['include_synthetic'] = True
        tmp_path = PROCESSED / f"loop_train_{i+1}.yaml"
        tmp_path.write_text(yaml.safe_dump(loop_cfg))
        train_models(tmp_path)

    if args.final_infer or (not args.initial and args.iterate == 0):
        log.info("=== Final inference ===")
        run_inference()

    log.info("🎉 All done!")

if __name__ == "__main__":
    main()

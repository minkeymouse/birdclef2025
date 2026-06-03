#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/birdclef26}"
cd "$ROOT"

python -m sidecar_src.datasets.make_folds \
  --data-dir data/birdclef-2026 \
  --out cache/folds.csv \
  --n-folds 5

python -m sidecar_src.datasets.build_metadata \
  --data-dir data/birdclef-2026 \
  --folds cache/folds.csv \
  --out-dir cache

python -m sidecar_src.training.train \
  --config sidecar_src/configs/exp001_pcen_convnext.yaml \
  --fold 0 \
  --epochs 1 \
  --limit-train 512 \
  --limit-valid 256 \
  --no-pretrained


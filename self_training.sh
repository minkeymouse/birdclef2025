#!/usr/bin/env bash
set -euo pipefail

#
# run_self_training.sh
#
# Usage:
#   ./run_self_training.sh [NUM_ITERS]
#
# If you pass a number as the first argument, it will run that many
# self-training iterations (default: 3).
#

NUM_ITERS="${1:-3}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python3"

echo "=========================================="
echo " Self-Training Pipeline"
echo " Project root:    $BASE_DIR"
echo " Iterations:      $NUM_ITERS"
echo "=========================================="
echo

# ── STEP 1: Initial melspec generation ───────────────────────────────────────
echo "[0] Generating initial mel-spectrogram chunks"
$PYTHON "$BASE_DIR/process_mels.py"
echo

# ── STEP 2: Initial rare-species weight bump ────────────────────────────────
echo "[1] Bumping rare-species weights"
$PYTHON "$BASE_DIR/weight_update.py"
echo

# ── STEP 3: Initial model training (pretrained) ─────────────────────────────
echo "[2] Training EfficientNet (pretrained)"
$PYTHON "$BASE_DIR/train_efficientnet.py" --pretrained
echo "[3] Training RegNetY (pretrained)"
$PYTHON "$BASE_DIR/train_regnety.py" --pretrained
echo

# ── STEP 4: Prepare unlabeled soundscapes ───────────────────────────────────
echo "[4] Initializing soundscape chunks"
$PYTHON "$BASE_DIR/initialize_soundscape.py"
echo

# ── STEP 5: Pseudo-label soundscapes ────────────────────────────────────────
echo "[5] Pseudo-labeling soundscapes"
$PYTHON "$BASE_DIR/inference.py"
echo

# ── SELF-TRAINING LOOP ───────────────────────────────────────────────────────
for ITER in $(seq 1 $NUM_ITERS); do
  echo "------------------------------------------"
  echo " Self-Training Iteration: $ITER/$NUM_ITERS"
  echo "------------------------------------------"

  echo "→ Bump rare-species weights"
  $PYTHON "$BASE_DIR/weight_update.py"
  echo

  echo "→ Re-training EfficientNet"
  $PYTHON "$BASE_DIR/train_efficientnet.py"
  echo

  echo "→ Re-training RegNetY"
  $PYTHON "$BASE_DIR/train_regnety.py"
  echo

  echo "→ Pseudo-labeling soundscapes"
  $PYTHON "$BASE_DIR/inference.py"
  echo
done

echo "=========================================="
echo "🎉 Self-training complete!"
echo "=========================================="

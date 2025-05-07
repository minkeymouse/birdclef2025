#!/usr/bin/env bash
set -euo pipefail

# run_self_training.sh
# Usage: ./run_self_training.sh [NUM_ITERS]
NUM_ITERS="${1:-3}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(which python3)"
CONDA_ENV="birdclef"

echo "=========================================="
echo " Self-Training Pipeline"
echo " Project root:    $BASE_DIR"
echo " Iterations:      $NUM_ITERS"
echo "=========================================="
echo

# ── STEP 0: Ensure tmux is installed ────────────────────────────────────────
if ! command -v tmux &> /dev/null; then
  echo "Error: tmux is required for parallel training." >&2
  exit 1
fi

# ── STEP 1: Activate conda environment ─────────────────────────────────────
echo "Activating conda env: $CONDA_ENV"
# shellcheck disable=SC1090
source ~/.bashrc
conda activate "$CONDA_ENV"
echo

# ── STEP 2: Initial melspec generation ───────────────────────────────────────
echo "[1] Generating initial mel-spectrogram chunks"
$PYTHON "$BASE_DIR/process_mels.py"
echo

# ── STEP 3: Initial rare-species weight bump ────────────────────────────────
echo "[2] Bumping rare-species weights"
$PYTHON "$BASE_DIR/weight_update.py"
echo

# ── STEP 4: Initial model training (pretrained) in parallel ─────────────────
echo "[3] Training EfficientNet & RegNetY (pretrained) in parallel"
# EfficientNet
tmux new-session -d -s eff_init bash -lc "source ~/.bashrc && conda activate $CONDA_ENV && \
  $PYTHON '$BASE_DIR/train_efficientnet.py' --pretrained && tmux wait-for -S eff_init_done"
# RegNetY
tmux new-session -d -s reg_init bash -lc "source ~/.bashrc && conda activate $CONDA_ENV && \
  $PYTHON '$BASE_DIR/train_regnety.py' --pretrained && tmux wait-for -S reg_init_done"

echo "→ Waiting for pretrained runs to finish..."
tmux wait-for eff_init_done
tmux wait-for reg_init_done
# cleanup
tmux kill-session -t eff_init
tmux kill-session -t reg_init
echo "→ Pretrained training complete"
echo

# ── STEP 5: Prepare unlabeled soundscapes ───────────────────────────────────
echo "[4] Initializing soundscape chunks"
$PYTHON "$BASE_DIR/initial_soundscapes.py"
echo

# ── STEP 6: Pseudo-label soundscapes ───────────────────────────────────────
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

  echo "→ Re-training EfficientNet & RegNetY in parallel"
  # EfficientNet
  tmux new-session -d -s eff_loop bash -lc "source ~/.bashrc && conda activate $CONDA_ENV && \
    $PYTHON '$BASE_DIR/train_efficientnet.py' && tmux wait-for -S eff_loop_done"
  # RegNetY
  tmux new-session -d -s reg_loop bash -lc "source ~/.bashrc && conda activate $CONDA_ENV && \
    $PYTHON '$BASE_DIR/train_regnety.py' && tmux wait-for -S reg_loop_done"

  echo "→ Waiting for retraining to finish..."
  tmux wait-for eff_loop_done
tmux wait-for reg_loop_done
  tmux kill-session -t eff_loop
tmux kill-session -t reg_loop
  echo "→ Retraining complete"
  echo

  echo "→ Pseudo-labeling soundscapes"
  $PYTHON "$BASE_DIR/inference.py"
  echo

done

echo "=========================================="
echo "🎉 Self-training complete!"
echo "=========================================="

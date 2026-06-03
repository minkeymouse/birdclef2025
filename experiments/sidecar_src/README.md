# BirdCLEF+ 2026 Sidecar SED

This package is a RunPod-oriented training pipeline for a custom sidecar model.
The base target is a compact 10-second context SED model:

```text
official train_soundscapes
→ fully labeled files only by default
→ 12 five-second windows per file
→ 2-channel [log-mel, PCEN] image
→ ConvNeXt/TIMM multilabel classifier
→ fold checkpoint + OOF predictions
→ Kaggle asset with infer.py
```

The model is meant to be blended into the current high-scoring anchor with a
masked rank correction, not to replace the anchor.

## Experiment Line

| Experiment | Training data | Purpose |
|---|---|---|
| `exp001_pcen_convnext_10s` | fully labeled train soundscapes only | small sanity sidecar |
| `exp002_pcen_convnext_10s_weakaudio` | train soundscapes + weakly labeled `train_audio` | broader species coverage with 10s context |
| `exp002b_pcen_convnext_5s_weakaudio` | train soundscapes + weakly labeled `train_audio` | sharper 5s label alignment and faster inference |

## RunPod Quick Start

From `/workspace/birdclef26`:

```bash
pip install -r sidecar_src/requirements-runpod.txt

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
  --fold 0
```

The exported Kaggle asset does not include dependency wheels. Install the
sidecar runtime dependencies in the notebook/session before running sidecar
inference:

```bash
pip install timm safetensors huggingface_hub
```

For a fast wiring check, add:

```bash
  --limit-train 512 --limit-valid 256 --epochs 1
```

After training all desired folds:

```bash
python -m sidecar_src.training.collect_oof \
  --config sidecar_src/configs/exp001_pcen_convnext.yaml \
  --folds 0 1 2 3 4
```

## OOF Gate

Global sidecar blending can cancel useful classes with noisy classes. Build a
class-wise OOF gate after collecting OOF predictions:

```bash
python -m sidecar_src.analysis.make_oof_gate \
  --oof outputs/exp002b_pcen_convnext_5s_weakaudio/oof_predictions.npz \
  --taxonomy data/birdclef-2026/taxonomy.csv \
  --sample data/birdclef-2026/sample_submission.csv \
  --out outputs/exp002b_pcen_convnext_5s_weakaudio/exp002b_oof_gate.csv
```

The final notebook reads `gate_weight` from this CSV and applies the sidecar as
a class-wise masked rank correction.

## Asset Export

After training at least one fold:

```bash
python -m sidecar_src.package_asset \
  --config sidecar_src/configs/exp001_pcen_convnext.yaml \
  --out-dir assets/sidecar_exp001 \
  --folds 0
```

For an exp002b asset with an OOF gate:

```bash
python -m sidecar_src.package_asset \
  --config sidecar_src/configs/exp002b_pcen_convnext_5s_weakaudio.yaml \
  --out-dir assets/sidecar_exp002b \
  --folds 0 \
  --gate-csv outputs/exp002b_pcen_convnext_5s_weakaudio/exp002b_oof_gate.csv
```

Attach the resulting directory as a private Kaggle Dataset, then call
`infer.py` from the final notebook to write `submission_sidecar.csv`.

Example final-notebook call:

```bash
python /kaggle/input/<sidecar-dataset>/infer.py \
  --data-dir /kaggle/input/birdclef-2026 \
  --input-dir /kaggle/input/birdclef-2026/test_soundscapes \
  --checkpoint-dir /kaggle/input/<sidecar-dataset> \
  --output submission_sidecar.csv \
  --device cpu \
  --folds 0
```

`infer.py` performs file-level inference: each 60-second soundscape is decoded
and resampled once, then split into 12 feature windows. This avoids the slow
row-level pattern of decoding the same file 12 times.

## exp002: Soundscape + Weak Train Audio

`exp002` keeps validation on official train-soundscape windows, but adds
balanced `train_audio` clips to the training side.  Primary labels are positive,
secondary labels are soft positives, and unknown negatives receive a low loss
mask so focal recordings do not become overconfident negative examples for all
other species.

Build metadata:

```bash
python -m sidecar_src.datasets.make_folds \
  --data-dir /tmp/birdclef_data/birdclef-2026 \
  --out cache_exp002/folds.csv \
  --n-folds 5

python -m sidecar_src.datasets.build_metadata \
  --data-dir /tmp/birdclef_data/birdclef-2026 \
  --folds cache_exp002/folds.csv \
  --out-dir cache_exp002 \
  --context-seconds 10 \
  --target-seconds 5

python -m sidecar_src.datasets.build_train_audio_metadata \
  --data-dir /tmp/birdclef_data/birdclef-2026 \
  --out-dir cache_exp002 \
  --n-folds 5 \
  --max-per-class 80 \
  --segments-per-file 1 \
  --secondary-weight 0.5 \
  --negative-mask-weight 0.15 \
  --context-seconds 10
```

Smoke check:

```bash
python -m sidecar_src.training.train \
  --config sidecar_src/configs/exp002_pcen_convnext_weakaudio.yaml \
  --fold 0 \
  --epochs 1 \
  --limit-train 512 \
  --limit-valid 128 \
  --no-pretrained
```

Full fold:

```bash
python -m sidecar_src.training.train \
  --config sidecar_src/configs/exp002_pcen_convnext_weakaudio.yaml \
  --fold 0
```

Package after training the desired folds:

```bash
python -m sidecar_src.package_asset \
  --config sidecar_src/configs/exp002_pcen_convnext_weakaudio.yaml \
  --out-dir assets/sidecar_exp002 \
  --folds 0
```

### exp002b 5s Variant

If `exp001/exp002` corrections look noisy, train the 5-second variant.  It
removes the 10s-context label mismatch and uses a smaller time image for faster
Kaggle CPU inference.

```bash
python -m sidecar_src.datasets.make_folds \
  --data-dir /tmp/birdclef_data/birdclef-2026 \
  --out cache_exp002b_5s/folds.csv \
  --n-folds 5

python -m sidecar_src.datasets.build_metadata \
  --data-dir /tmp/birdclef_data/birdclef-2026 \
  --folds cache_exp002b_5s/folds.csv \
  --out-dir cache_exp002b_5s \
  --context-seconds 5 \
  --target-seconds 5

python -m sidecar_src.datasets.build_train_audio_metadata \
  --data-dir /tmp/birdclef_data/birdclef-2026 \
  --out-dir cache_exp002b_5s \
  --n-folds 5 \
  --max-per-class 60 \
  --segments-per-file 1 \
  --secondary-weight 0.5 \
  --negative-mask-weight 0.10 \
  --context-seconds 5

python -m sidecar_src.training.train \
  --config sidecar_src/configs/exp002b_pcen_convnext_5s_weakaudio.yaml \
  --fold 0
```

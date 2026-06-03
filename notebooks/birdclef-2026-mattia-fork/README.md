# birdclef-2026-mattia-fork — PRODUCTION (LB 0.949)

Current notebook = **kojimar 0.949 fork** (`kojimar/0-949-lb-birdclef-2026-prior-axis-rank-fusion`).
All public components: Tucker SED + Imaad pipeline + yukiZ ProtoSSM.

## Lineage

1. 2026-05-03: forked Mattia 0.941 pipeline → v5 = 0.938
2. 2026-05-15: swapped to mtoshi_947 (Imaad v4) → v53 = 0.943
3. 2026-05-21: swapped to kojimar 0.949 (Prior-Axis Rank Fusion) → **0.949 current**

Own-weights variant (v_kojimar_lever1 = 0.945): kojimar + own SED [0..4,15..19] + xSED=[0.70, 0.30].

## Files

- `notebook.ipynb` — Kaggle kernel. Pushed as `ultimatumgame/birdclef-2026-mattia-fork`.
- `kernel-metadata.json` — Kaggle input datasets.

## Push / Submit

```bash
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) uv run kaggle kernels push -p notebooks/birdclef-2026-mattia-fork
uv run python -m experiments.sed.one_shot_submit --version <N> --message "..."
```

## Kaggle inputs

- `birdclef-2026` (competition data)
- `rishikeshjani/perch-onnx-for-birdclef-2026` — Perch ONNX
- `tuckerarrants/bc2026-distilled-sed-public` — Tucker public SED
- `tuckerarrants/perch-v2-no-dft-onnx` — Perch v2
- `hideyukizushi/sgkfk-202604041716` — yukiZ ProtoSSM weights
- `jaejohn/perch-meta` — Perch metadata cache

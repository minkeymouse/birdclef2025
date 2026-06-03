# LB results summary

- **v28** (2026-05-08 03:09): LB 0.8370 (Δ -0.1010) — Q3: drop rare classes n_train_audio<5 (42 cls). Test if rare-supervision classes are net positive.
- **v27** (2026-05-08 03:09): LB 0.8340 (Δ -0.1040) — Q2: drop all Amphibia (35 cls). Same logic as Q1 for Amphibia taxon.
- **v26** (2026-05-08 03:09): LB 0.8810 (Δ -0.0570) — Q1: drop all Insecta (28 cls)
- **v30** (2026-05-08 05:43): LB 0.9370 (Δ -0.0010) — exp175 5-fold seed=42 (silent drift fix: drop_rate removed + xavier init). 5-fold mean val_SS 0.8689
- **v32** (2026-05-08 10:45): LB 0.9370 (Δ -0.0010) — exp176 5-fold seed=42 (per-fold SS split, Tucker-correct EVAL_SS_N_FILES=None). 5-fold mean val_SS 0

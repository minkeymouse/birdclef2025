# CLEF 2026 working note — paper artifacts

**Thesis:** covariate shift in cross-region bioacoustic monitoring — local validation (single-site-dominated)
decouples from cross-region test performance, so the in-distribution SOTA recipe (better SED + external data +
post-proc) does not transfer. Comp deadline 2026-06-03; paper 2026-06-17.

## Read in this order
1. **`covariate_shift_findings_2026_06_03.md`** — MASTER findings + evidence (the consolidated working-note
   content). Findings 1–4 + the exp187/exp189 case studies + the transfer-decoupling table + next directions.
2. `structural_analysis_2026_06_02.md` — supporting data: evaluability by taxon, site-22 distribution (65% of
   labeled SS), the untrainable Insecta sonotypes, cold-start species.
3. `next_directions_2026_06_02.md` — deep lit-research synthesis (foundation models / domain adaptation /
   semi-supervised / few-shot); the candidate menu and the AUC-invariance reframe.

## Drafts (LaTeX)
- `exp_current.tex` — current full draft.
- `experiments.tex` — experiments section.
- `eda.tex` — EDA section.

## Headline empirical result (the paper's core)
exp189 (Tucker SED recipe + external non-Aves data) **beats the production SED on every evaluable group**
locally (all-eval 0.9979 > 0.9972) yet is **LB-flat (0.950)**. A strictly-better-locally model moves the test
metric by 0.000 → isolates *transfer* from *capacity*. Across interventions, local Δ of +0.0006 to +0.067 all
produced ~0 LB movement (decoupling table in the master doc).

_Older planning docs: `_archive/`._

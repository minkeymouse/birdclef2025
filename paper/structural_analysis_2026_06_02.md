# BirdCLEF 2026 — Structural Analysis of the 0.950 Plateau

_Generated 2026-06-02. Corrected multi-label parse of `train_soundscapes_labels.csv` (`primary_label` is semicolon-separated). Supporting data for `covariate_shift_findings_2026_06_03.md` (the canonical working-note doc) — these are the raw source numbers; see README for read order._

> Note on thresholds: this doc counts evaluable = **≥1 TP window → 75 species** on the 1478-window multi-label
> substrate. The master doc's controlled eval uses the stricter **≥10 positives → 50 species** on its 739-window
> aligned substrate. Both are correct (different thresholds/substrates), not a discrepancy.

## 1. Evaluability by taxonomic class
Macro-AUC skips classes with no true positives. Of 234 species, **75 are evaluable** in the labeled soundscapes (≥1 TP window across 1478 windows / 66 files).

| Class | total | evaluable | TP windows | Perch-mapped | train_audio |
|---|---|---|---|---|---|
| Aves | 162 | 28 | 824 | 162/162 | 162/162 |
| Insecta | 28 | 25 | 1136 | 0/28 | 3/28 |
| Amphibia | 35 | 17 | 4174 | 32/35 | 32/35 |
| Mammalia | 8 | 4 | 84 | 6/8 | 8/8 |
| Reptilia | 1 | 1 | 26 | 0/1 | 1/1 |

## 2. Site covariate shift (the core thesis)
- Labeled soundscapes are dominated by **site S22: 954/1478 windows (65%), 40/66 files**.
- Full site distribution (windows): S22=954, S08=120, S15=96, S19=72, S23=72, S13=48, S03=48, S09=38, S18=30.
- **Test set is site S05** (`BC2026_Test_0001_S05_20250227_...`), which does **not** appear in the labeled set.
- Consequence: any per-class calibration fit on labeled SS fits S22 acoustics and **anti-correlates** with the unseen-site test (empirically confirmed: per-class blend file-CV +0.046 → site-LOSO −0.016).

## 3. The untrainable Insecta sonotypes
- The 25 `47158son*` sonotypes (call-types of one insect) are **all evaluable** (TP 6–168, total 1136 windows) but have **0 train_audio and 0 external** recordings → untrainable by supervised means.
- They co-occur densely: of 336 windows with ≥1 sonotype, mean **3.38** sonotypes/window, 74% have >1. But each sonotype is sparse among the co-occurring set, so a max-pool 'mirror' post-proc would inject false positives → not a clean lever.
- Cold-start species overall: **28** (0 train_audio) = 25 Insecta + 3 Amphibia.

## 4. Why the plateau holds (mechanism)
Every evaluable species falls into one of two already-closed buckets, under a calibration trap:
1. **Already near-optimally handled** — evaluable Aves (Perch 162/162) and Amphibia (Perch 32/35) are well-served by the public Perch+ProtoSSM+SED pipeline; self-trained additions are weaker than Tucker and redundant (verified: R1 train_audio Perch probe on high-TP Amphibia = LB 0.950 flat).
2. **Untrainable but SED-handled** — the 25 sonotypes have no supervised data yet are scored ~0.85 AUC by the distilled SED learned from soundscape labels; improving them needs test-distribution data we lack.
3. **Calibration trap** — the only labeled soundscape data is 65% S22 while the test is unseen S05, so label-fitted per-class corrections anti-correlate with the hidden test.

This explains why independent public pipelines (EoS.9, Karnakbayev, ours) all converge to LB ≈ 0.950.

> ⚠ UPDATE 2026-06-03: the "closing the gap requires the 2025-winner components (external data, multi-round
> Noisy Student / multi-teacher distill)" hypothesis was tested and REFUTED. exp189 (external non-Aves data,
> beat Tucker on every evaluable group) was LB-flat (0.950); exp187 (heterogeneous multi-teacher distill +
> SoftAUC, the winner recipe) was LB 0.938. Even a strictly-better-locally SED moved the LB by 0.000 → the
> operative ceiling is local→LB TRANSFER (this §3 calibration trap), NOT the missing data/recipe components.
> See `covariate_shift_findings_2026_06_03.md` Findings 3–3c.

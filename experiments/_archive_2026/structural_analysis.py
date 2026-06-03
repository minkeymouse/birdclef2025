"""Structural analysis of the BirdCLEF 2026 0.950 plateau (CORRECTED multi-label parse).
Quantifies: (1) evaluability by class, (2) site covariate shift, (3) sonotype structure,
(4) Perch coverage. Emits paper/structural_analysis_2026_06_02.md. No LB slots used."""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path("/data/birdclef2026"); DATA = ROOT / "data/birdclef-2026"
lab = pd.read_csv(DATA / "train_soundscapes_labels.csv")
tax = pd.read_csv(DATA / "taxonomy.csv")
train = pd.read_csv(DATA / "train.csv")
cls = tax.set_index(tax["primary_label"].astype(str))["class_name"].to_dict()

# --- evaluability (multi-label ';' parse) ---
lab["splist"] = lab["primary_label"].astype(str).str.split(";")
ex = lab.explode("splist"); ex["splist"] = ex["splist"].str.strip()
ex = ex[ex["splist"] != ""]
tp = ex["splist"].value_counts()
ev_by_cls = {}
for s, n in tp.items():
    ev_by_cls.setdefault(cls.get(s, "?"), []).append((s, int(n)))

# --- trainable (train_audio) coverage ---
train_sp = set(train["primary_label"].astype(str))
all_sp = set(tax["primary_label"].astype(str))
coldstart = all_sp - train_sp

# --- site distribution (covariate shift) ---
lab["site"] = lab["filename"].str.extract(r"_(S\d+)_")
site_win = lab["site"].value_counts()
site_file = lab.drop_duplicates("filename")["site"].value_counts()

# --- sonotype structure ---
son = [s for s in all_sp if s.startswith("47158son")]
son_tp = {s: int(tp.get(s, 0)) for s in son}
son_set = set(son)
lab["n_son"] = lab["splist"].apply(lambda L: len(son_set & set(x.strip() for x in L)))
sonwin = lab[lab["n_son"] > 0]

# --- Perch mapping coverage (from project docs CLAUDE.md/memory) ---
perch_map = {"Aves": "162/162", "Insecta": "0/28", "Amphibia": "32/35", "Mammalia": "6/8", "Reptilia": "0/1"}

R = []
R.append("# BirdCLEF 2026 — Structural Analysis of the 0.950 Plateau\n")
R.append("_Generated 2026-06-02. Corrected multi-label parse of `train_soundscapes_labels.csv` "
         "(`primary_label` is semicolon-separated). Companion data for the CLEF working note "
         "(thesis: covariate shift in cross-region bioacoustic monitoring)._\n")

R.append("## 1. Evaluability by taxonomic class")
R.append("Macro-AUC skips classes with no true positives. Of 234 species, **%d are evaluable** "
         "in the labeled soundscapes (have ≥1 TP window across %d windows / %d files).\n"
         % (int(ex["splist"].nunique()), len(lab), lab["filename"].nunique()))
R.append("| Class | total | evaluable | TP windows | Perch-mapped | train_audio |")
R.append("|---|---|---|---|---|---|")
for c in ["Aves", "Insecta", "Amphibia", "Mammalia", "Reptilia"]:
    lst = ev_by_cls.get(c, [])
    n_total = int((tax["class_name"] == c).sum())
    n_train = sum(1 for s in all_sp if cls.get(s) == c and s in train_sp)
    R.append(f"| {c} | {n_total} | {len(lst)} | {sum(n for _,n in lst)} | {perch_map.get(c,'?')} | {n_train}/{n_total} |")
R.append("")

R.append("## 2. Site covariate shift (the core thesis)")
R.append(f"- Labeled soundscapes are dominated by **site S22: {int(site_win.get('S22',0))}/{len(lab)} "
         f"windows ({100*site_win.get('S22',0)/len(lab):.0f}%), {int(site_file.get('S22',0))}/{lab['filename'].nunique()} files**.")
R.append(f"- Full site distribution (windows): " + ", ".join(f"{s}={int(n)}" for s, n in site_win.items()) + ".")
R.append("- **Test set is site S05** (`BC2026_Test_0001_S05_20250227_...`), which does **not** appear in the labeled set.")
R.append("- Consequence: any per-class calibration fit on labeled SS fits S22 acoustics and **anti-correlates** "
         "with the unseen-site test (empirically confirmed: per-class blend file-CV +0.046 → site-LOSO −0.016).\n")

R.append("## 3. The untrainable Insecta sonotypes")
R.append(f"- The 25 `47158son*` sonotypes (call-types of one insect) are **all evaluable** "
         f"(TP {min(son_tp.values())}–{max(son_tp.values())}, total {sum(son_tp.values())} windows) "
         f"but have **0 train_audio and 0 external** recordings → untrainable by supervised means.")
R.append(f"- They co-occur densely: of {len(sonwin)} windows with ≥1 sonotype, mean **{sonwin['n_son'].mean():.2f}** "
         f"sonotypes/window, {100*(sonwin['n_son']>1).mean():.0f}% have >1. But each sonotype is sparse among the "
         f"co-occurring set, so a max-pool 'mirror' post-proc would inject false positives → not a clean lever.")
R.append(f"- Cold-start species overall: **{len(coldstart)}** (0 train_audio) = {sum(1 for s in coldstart if cls.get(s)=='Insecta')} Insecta + "
         f"{sum(1 for s in coldstart if cls.get(s)=='Amphibia')} Amphibia.\n")

R.append("## 4. Why the plateau holds (mechanism)")
R.append("Every evaluable species falls into one of two already-closed buckets, under a calibration trap:")
R.append("1. **Already near-optimally handled** — evaluable Aves (Perch 162/162) and Amphibia (Perch 32/35) are "
         "well-served by the public Perch+ProtoSSM+SED pipeline; self-trained additions are weaker than Tucker "
         "and redundant (verified: R1 train_audio Perch probe on high-TP Amphibia = LB 0.950 flat).")
R.append("2. **Untrainable but SED-handled** — the 25 sonotypes have no supervised data yet are scored ~0.85 AUC "
         "by the distilled SED learned from soundscape labels; improving them needs test-distribution data we lack.")
R.append("3. **Calibration trap** — the only labeled soundscape data is 65% S22 while the test is unseen S05, so "
         "label-fitted per-class corrections anti-correlate with the hidden test.")
R.append("\nThis explains why independent public pipelines (EoS.9, Karnakbayev, ours) all converge to LB ≈ 0.950. "
         "Closing the gap requires the components the 2025 winner had and this line lacks "
         "(dedicated extended-species models, multi-round Noisy Student, large external pretraining) — multi-day work, not inference levers.\n")

out = ROOT / "paper/structural_analysis_2026_06_02.md"
out.write_text("\n".join(R))
print(f"wrote {out} ({len(R)} lines)")
print("\n".join(R[:60]))

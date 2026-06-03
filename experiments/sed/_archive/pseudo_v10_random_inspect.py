"""Manual inspection — random 30 sample windows from v10.

User asked: "윈도우 하나하나 보고 판단해도 OK"
We can't listen to audio in this environment, but we can:
  - Cross-check multi-criteria scores per window
  - Verify cooccurrence patterns make ecological sense
  - Flag suspicious entries (e.g., 4 sonotypes from different families in one window)

Outputs a markdown report with 30 examples sampled stratified by tier.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
V10 = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v10.csv"
TAXONOMY = ROOT / "data/birdclef-2026/taxonomy.csv"
TRAIN_CSV = ROOT / "data/birdclef-2026/train.csv"
OUT = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v10_inspect.md"


def main():
    v10 = pd.read_csv(V10)
    tax = pd.read_csv(TAXONOMY).set_index("primary_label")
    train = pd.read_csv(TRAIN_CSV)
    n_train = train.groupby("primary_label").size().to_dict()

    rng = np.random.RandomState(2026517)

    # Build window-level summary: for each (filename, end) group all labels
    grouped = v10.groupby(["filename", "start", "end"])
    windows = []
    for (fn, start, end), grp in grouped:
        labs = grp["primary_label"].tolist()
        tiers = grp["tier"].tolist()
        scores = list(zip(
            grp["v33_score"].fillna(0).tolist(),
            grp["perch_score"].fillna(0).tolist(),
            grp["emb_sim"].fillna(0).tolist(),
            grp["cos_sim"].fillna(0).tolist(),
        ))
        sources = grp["source"].tolist()
        site = grp["filename"].iloc[0].split("_")[3]  # SXX
        windows.append({
            "filename": fn, "site": site, "start": start, "end": end,
            "labels": labs, "tiers": tiers, "scores": scores, "sources": sources,
            "n_labels": len(labs),
        })

    print(f"Total windows: {len(windows)}")

    # Stratify samples
    # Tier-A only windows: 10
    # Mixed v2+sonotype windows: 10
    # Sonotype-only windows: 10
    tier_a_only = [w for w in windows if all(t == "A_common" for t in w["tiers"])]
    mixed = [w for w in windows if "A_common" in w["tiers"] and "C_ultra_sonotype" in w["tiers"]]
    son_only = [w for w in windows if all(t == "C_ultra_sonotype" for t in w["tiers"])]

    sample_a = rng.choice(len(tier_a_only), min(10, len(tier_a_only)), replace=False)
    sample_m = rng.choice(len(mixed), min(10, len(mixed)), replace=False)
    sample_s = rng.choice(len(son_only), min(10, len(son_only)), replace=False)

    lines = []
    lines.append("# v10 Random window inspection (n=30, stratified by tier mix)\n\n")
    lines.append("Generated 2026-05-17 sprint Phase F. Cross-check whether v10 row-level decisions are reasonable.\n\n")

    def emit_window(w, label):
        out = []
        out.append(f"## {label}\n")
        out.append(f"**file**: `{w['filename']}` | **site**: {w['site']} | **window**: {w['start']}-{w['end']}s | **n_labels**: {w['n_labels']}\n")
        out.append(f"\n| species | n_train | taxon | tier | source | v33_score | perch_score | emb_sim | cos_sim |\n")
        out.append("|---|---|---|---|---|---|---|---|---|\n")
        for lab, tier, sc, src in zip(w["labels"], w["tiers"], w["scores"], w["sources"]):
            ntr = n_train.get(lab, 0)
            tx = tax.loc[lab, "class_name"] if lab in tax.index else "?"
            v33, prc, emb, cos = sc
            v33s = f"{v33:.3f}" if v33 > 0 else "—"
            prcs = f"{prc:.3f}" if prc > 0 else "—"
            embs = f"{emb:.3f}" if emb > 0 else "—"
            coss = f"{cos:.3f}" if cos > 0 else "—"
            out.append(f"| `{lab}` | {ntr} | {tx} | {tier} | {src} | {v33s} | {prcs} | {embs} | {coss} |\n")
        # Cross-row consistency check
        taxons = [tax.loc[l, "class_name"] if l in tax.index else "?" for l in w["labels"]]
        unique_taxa = set(taxons)
        out.append(f"\n**Inspection notes**: ")
        if len(unique_taxa) == 1:
            out.append(f"All {len(w['labels'])} labels are {next(iter(unique_taxa))} — taxonomically coherent. ")
        else:
            out.append(f"{len(unique_taxa)} different taxa in same window: {sorted(unique_taxa)} — multi-taxa cooccurrence. ")

        # Check if sonotypes are from same family (47158son*)
        son_labs = [l for l in w["labels"] if l.startswith("47158son")]
        if len(son_labs) >= 2:
            son_ids = [int(l.replace("47158son", "")) for l in son_labs]
            if max(son_ids) - min(son_ids) <= 5:
                out.append(f"Sonotypes ({son_labs}) appear to be close-numbered (similar sonogram family). ")
            else:
                out.append(f"Sonotypes ({son_labs}) span wide numerical range — verify they're really co-firing. ")
        out.append("\n\n---\n\n")
        return "".join(out)

    for i, idx in enumerate(sample_a):
        lines.append(emit_window(tier_a_only[idx], f"Tier-A only #{i+1}"))
    for i, idx in enumerate(sample_m):
        lines.append(emit_window(mixed[idx], f"Mixed (A + Sonotype) #{i+1}"))
    for i, idx in enumerate(sample_s):
        lines.append(emit_window(son_only[idx], f"Sonotype only #{i+1}"))

    OUT.write_text("".join(lines))
    print(f"Saved: {OUT}")
    print(f"Length: {len(''.join(lines))} chars")


if __name__ == "__main__":
    main()

"""Pseudo v10 v2 refine — Tier B threshold loosen + window inspection.

Phase F continued: v10 first cut showed Tier B (5 ≤ n_train < 20) drop 99.7%
which is too aggressive. Loosen criterion.

Also: inspect window-by-window for sample (file × window × class):
  - Are Tier C kept rows really high-confidence sonotypes?
  - Are Tier B kept rows agreeing across teachers?
  - Sample diagnostic of pseudo label quality.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))

V7 = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v7.csv"
TRAIN_CSV = ROOT / "data/birdclef-2026/train.csv"
TAXONOMY_CSV = ROOT / "data/birdclef-2026/taxonomy.csv"
OUT_V10 = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v10.csv"


def main():
    print("Loading...")
    v7 = pd.read_csv(V7)
    train = pd.read_csv(TRAIN_CSV)
    tax = pd.read_csv(TAXONOMY_CSV)

    n_train = train.groupby("primary_label").size().to_dict()
    sp_to_tax = dict(zip(tax.primary_label.astype(str), tax.class_name))

    v7["n_train"] = v7["primary_label"].map(n_train).fillna(0).astype(int)
    v7["taxon_class"] = v7["primary_label"].map(sp_to_tax).fillna("?")

    # === Refined filter ===
    keep_mask = pd.Series(False, index=v7.index)

    # Tier A: common (n_train ≥ 20) — loose, accept many
    tier_a = v7["n_train"] >= 20
    keep_a = tier_a & ((v7["v33_score"] >= 0.55) | (v7["perch_score"] >= 0.25) | (v7["exp50_score"] >= 0.3))
    keep_mask |= keep_a

    # Tier B: rare (5 ≤ n_train < 20) — LOOSENED from 0.7 to 0.6, OR-based
    tier_b = (v7["n_train"] >= 5) & (v7["n_train"] < 20)
    # accept if v33 ≥ 0.6 (single teacher confidence)
    # OR perch ≥ 0.3 (Perch teacher agreement)
    # OR (emb_sim ≥ 0.5 AND v33 ≥ 0.5) (embedding similarity to centroid + moderate score)
    keep_b = tier_b & (
        (v7["v33_score"] >= 0.6) |
        (v7["perch_score"] >= 0.3) |
        ((v7["emb_sim"] >= 0.5) & (v7["v33_score"] >= 0.5))
    )
    keep_mask |= keep_b

    # Tier C: ultra-rare (n_train < 5)
    tier_c = v7["n_train"] < 5
    # v2 source: v33_score is essentially 0 here; rely on perch_score (rare to have)
    keep_c_v2 = tier_c & (v7["source"] == "v2") & (v7["perch_score"] >= 0.4)
    # sonotype source: cosine-similarity-based; require strong agreement
    keep_c_son = tier_c & (v7["source"] == "sonotype") & (v7["cos_sim"] >= 0.65)
    keep_c = keep_c_v2 | keep_c_son
    keep_mask |= keep_c

    # Statistics
    print(f"\n=== REFINED FILTER ===")
    print(f"  Tier A (n_train ≥ 20):       {tier_a.sum():>7d} rows → keep {keep_a.sum():>7d} ({100*keep_a.sum()/max(tier_a.sum(),1):.1f}%)")
    print(f"  Tier B (5 ≤ n_train < 20):  {tier_b.sum():>7d} rows → keep {keep_b.sum():>7d} ({100*keep_b.sum()/max(tier_b.sum(),1):.1f}%)")
    print(f"  Tier C (n_train < 5):       {tier_c.sum():>7d} rows → keep {keep_c.sum():>7d} ({100*keep_c.sum()/max(tier_c.sum(),1):.1f}%)")
    print(f"    Tier C/v2:                                  {keep_c_v2.sum():>7d}")
    print(f"    Tier C/sonotype:                            {keep_c_son.sum():>7d}")
    print(f"  TOTAL: {len(v7):>7d} → keep {keep_mask.sum():>7d} ({100*keep_mask.sum()/len(v7):.1f}%)")

    # Produce v10
    v10 = v7[keep_mask].copy().reset_index(drop=True)
    v10["tier"] = "?"
    v10.loc[v10["n_train"] >= 20, "tier"] = "A_common"
    v10.loc[(v10["n_train"] >= 5) & (v10["n_train"] < 20), "tier"] = "B_rare"
    v10.loc[(v10["n_train"] < 5) & (v10["source"] == "v2"), "tier"] = "C_ultra_v2"
    v10.loc[(v10["n_train"] < 5) & (v10["source"] == "sonotype"), "tier"] = "C_ultra_sonotype"

    print(f"\n=== Per-tier counts ===")
    print(v10["tier"].value_counts())

    # Per-class change
    v10_by_cls = v10.groupby("primary_label").size().reset_index(name="rows_v10")
    v10_by_cls["n_train"] = v10_by_cls["primary_label"].map(n_train).fillna(0).astype(int)
    v10_by_cls["taxon"] = v10_by_cls["primary_label"].map(sp_to_tax).fillna("?")
    v7_by_cls = v7.groupby("primary_label").size().reset_index(name="rows_v7")
    cmp = v10_by_cls.merge(v7_by_cls, on="primary_label", how="left")
    cmp["drop_pct"] = 100 * (1 - cmp["rows_v10"] / cmp["rows_v7"])

    print(f"\n=== Top 15 rare/ultra-rare classes (n_train < 20) — kept vs dropped ===")
    cmp_rare = cmp[cmp["n_train"] < 20].sort_values("rows_v10", ascending=False)
    print(f"  {'species':<10}  {'tx':<8}  {'n_tr':>4}  {'v7':>7}  {'v10':>7}  {'drop%':>5}")
    for _, r in cmp_rare.head(15).iterrows():
        print(f"  {r['primary_label']:<10}  {r['taxon']:<8}  {r['n_train']:>4d}  {r['rows_v7']:>7d}  {r['rows_v10']:>7d}  {r['drop_pct']:>5.1f}%")

    # === Window-by-window inspection — sample 20 rows from each tier ===
    print(f"\n=== WINDOW-BY-WINDOW SAMPLE INSPECTION ===\n")
    rng = np.random.RandomState(42)
    for tier_name in ["A_common", "B_rare", "C_ultra_v2", "C_ultra_sonotype"]:
        sub = v10[v10["tier"] == tier_name]
        if len(sub) == 0:
            continue
        n_sample = min(5, len(sub))
        idx = rng.choice(len(sub), n_sample, replace=False)
        print(f"\n--- Tier {tier_name} sample (n={n_sample} of {len(sub)} total) ---")
        print(f"  {'filename':<35}  {'start':>5}  {'end':>4}  {'label':<10}  {'tx':<8}  {'v33':>5}  {'perch':>5}  {'emb_sim':>7}  {'cos_sim':>7}  {'src':<8}")
        for i in idx:
            r = sub.iloc[i]
            v33 = f"{r['v33_score']:.3f}" if pd.notna(r['v33_score']) else "n/a"
            perch = f"{r['perch_score']:.3f}" if pd.notna(r['perch_score']) else "n/a"
            emb = f"{r['emb_sim']:.3f}" if pd.notna(r['emb_sim']) else "n/a"
            cos = f"{r['cos_sim']:.3f}" if pd.notna(r['cos_sim']) else "n/a"
            src = r['source']
            print(f"  {r['filename']:<33}  {r['start']:>5}  {r['end']:>4}  {r['primary_label']:<10}  {r['taxon_class']:<8}  {v33:>5}  {perch:>5}  {emb:>7}  {cos:>7}  {src:<8}")

    # === Dropped sample inspection ===
    dropped = v7[~keep_mask].copy()
    dropped["tier_potential"] = "?"
    dropped.loc[dropped["n_train"] >= 20, "tier_potential"] = "A_common_dropped"
    dropped.loc[(dropped["n_train"] >= 5) & (dropped["n_train"] < 20), "tier_potential"] = "B_rare_dropped"
    dropped.loc[(dropped["n_train"] < 5) & (dropped["source"] == "v2"), "tier_potential"] = "C_v2_dropped"
    dropped.loc[(dropped["n_train"] < 5) & (dropped["source"] == "sonotype"), "tier_potential"] = "C_son_dropped"

    print(f"\n=== DROPPED rows — examples to verify filter correctness ===\n")
    for tier_name in ["A_common_dropped", "B_rare_dropped", "C_son_dropped"]:
        sub = dropped[dropped["tier_potential"] == tier_name]
        if len(sub) == 0: continue
        n_sample = min(5, len(sub))
        idx = rng.choice(len(sub), n_sample, replace=False)
        print(f"\n--- {tier_name} (dropped) sample (n={n_sample} of {len(sub)}) ---")
        print(f"  {'filename':<35}  {'start':>5}  {'end':>4}  {'label':<10}  {'v33':>5}  {'perch':>5}  {'emb_sim':>7}  {'cos_sim':>7}")
        for i in idx:
            r = sub.iloc[i]
            v33 = f"{r['v33_score']:.3f}" if pd.notna(r['v33_score']) else "n/a"
            perch = f"{r['perch_score']:.3f}" if pd.notna(r['perch_score']) else "n/a"
            emb = f"{r['emb_sim']:.3f}" if pd.notna(r['emb_sim']) else "n/a"
            cos = f"{r['cos_sim']:.3f}" if pd.notna(r['cos_sim']) else "n/a"
            print(f"  {r['filename']:<33}  {r['start']:>5}  {r['end']:>4}  {r['primary_label']:<10}  {v33:>5}  {perch:>5}  {emb:>7}  {cos:>7}")

    # Save
    v10.to_csv(OUT_V10, index=False)
    print(f"\nSaved v10 (refined): {OUT_V10}  ({len(v10)} rows)")

    # Save summary stats
    import json
    summary = {
        "v7_total_rows": int(len(v7)),
        "v10_total_rows": int(len(v10)),
        "v10_kept_pct": float(100*len(v10)/len(v7)),
        "tier_counts": v10["tier"].value_counts().to_dict(),
        "per_tier_filter_rule": {
            "A_common (n_train >= 20)": "v33≥0.55 OR perch≥0.25 OR exp50≥0.30",
            "B_rare (5 ≤ n_train < 20)": "v33≥0.60 OR perch≥0.30 OR (emb_sim≥0.5 AND v33≥0.50)",
            "C_ultra_v2 (n_train<5, v2 source)": "perch≥0.40",
            "C_ultra_sonotype (n_train<5, sonotype source)": "cos_sim≥0.65",
        }
    }
    json_path = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v10.json"
    json.dump(summary, json_path.open("w"), indent=2, default=str)
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()

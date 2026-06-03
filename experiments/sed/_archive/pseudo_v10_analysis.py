"""Pseudo label v10 — axis-2-aware filter design (2026-05-17 sprint).

Phase F: 2026-05-17 OPQHEF sprint finding axis-2 (ours over-predicts ultra-rare
in broader OOD pool) implies existing pseudo labels v3/v7/v8 have noise in
ultra-rare class rows (v33 teacher = ours, propagates axis-2 noise).

Analysis goal: design a multi-criteria filter that:
  1. Identifies high-noise rows in ultra-rare classes (n_train < 5)
  2. Keeps high-confidence rows in common classes (n_train ≥ 20)
  3. Uses v7's already-computed features (v33_score, exp50_score, perch_score, emb_sim, cos_sim)
  4. Does NOT require new SED inference (compute-bounded by 5h budget)

Output: per-class distribution stats + filter recipe + v10 produce.
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
    print(f"v7: {v7.shape}")

    n_train = train.groupby("primary_label").size().to_dict()
    sp_to_tax = dict(zip(tax.primary_label.astype(str), tax.class_name))

    v7["n_train"] = v7["primary_label"].map(n_train).fillna(0).astype(int)
    v7["taxon_class"] = v7["primary_label"].map(sp_to_tax).fillna("?")

    # === 1. Per-class row count vs n_train ===
    print("\n=== PER-CLASS row count (top 30 + ultra-rare focus) ===")
    by_cls = v7.groupby("primary_label").agg(
        rows=("filename", "size"),
        v33_mean=("v33_score", "mean"),
        v33_max=("v33_score", "max"),
        perch_mean=("perch_score", "mean"),
        perch_max=("perch_score", "max"),
        emb_sim_mean=("emb_sim", "mean"),
        cos_sim_mean=("cos_sim", "mean"),
    ).reset_index()
    by_cls["n_train"] = by_cls["primary_label"].map(n_train).fillna(0).astype(int)
    by_cls["taxon"] = by_cls["primary_label"].map(sp_to_tax).fillna("?")
    by_cls = by_cls.sort_values("rows", ascending=False)

    print(f"  {'species':<10}  {'tx':<8}  {'n_tr':>5}  {'rows':>7}  {'v33_mean':>9}  {'perch_mean':>11}")
    for _, r in by_cls.head(15).iterrows():
        print(f"  {r['primary_label']:<10}  {r['taxon']:<8}  {r['n_train']:>5d}  {r['rows']:>7d}  {r['v33_mean']:>9.3f}  {r['perch_mean']:>11.3f}")

    print(f"\n=== ULTRA-RARE (n_train < 5) class row distribution ===")
    ultra = by_cls[by_cls["n_train"] < 5].copy()
    ultra = ultra.sort_values("rows", ascending=False)
    print(f"  {'species':<10}  {'tx':<8}  {'n_tr':>5}  {'rows':>7}  {'v33_mean':>9}  {'perch_mean':>11}  {'emb_sim_mean':>13}  {'cos_sim_mean':>13}")
    total_ultra_rows = 0
    for _, r in ultra.iterrows():
        total_ultra_rows += r['rows']
        if r['rows'] >= 100:  # only meaningful classes
            print(f"  {r['primary_label']:<10}  {r['taxon']:<8}  {r['n_train']:>5d}  {r['rows']:>7d}  "
                  f"{r['v33_mean']:>9.3f}  {r['perch_mean']:>11.3f}  {r['emb_sim_mean']:>13.3f}  {r['cos_sim_mean']:>13.3f}")
    print(f"  Total ultra-rare rows: {total_ultra_rows} ({100*total_ultra_rows/len(v7):.1f}%)")

    # === 2. Multi-criteria score distribution: ultra-rare vs common ===
    print(f"\n=== Multi-criteria DISTRIBUTION: ultra-rare vs common ===")
    ur_rows = v7[v7["n_train"] < 5]
    cm_rows = v7[v7["n_train"] >= 20]
    print(f"  ultra-rare ({len(ur_rows)} rows) vs common ({len(cm_rows)} rows)")
    print(f"  {'metric':<15}  {'ultra-rare':>15}  {'common':>15}")
    for col in ["v33_score", "exp50_score", "perch_score", "emb_sim", "cos_sim"]:
        u_vals = ur_rows[col].dropna()
        c_vals = cm_rows[col].dropna()
        if len(u_vals) == 0 or len(c_vals) == 0:
            continue
        for q in [50, 75, 90]:
            up = u_vals.quantile(q/100)
            cp = c_vals.quantile(q/100)
            print(f"  {col:<10} p{q:>2}  {up:>15.4f}  {cp:>15.4f}")

    # === 3. Source breakdown ===
    print(f"\n=== Source counts ===")
    print(v7['source'].value_counts())

    print("\n=== Sonotype source (v7 has explicit sonotype source) ===")
    son = v7[v7['source'] == 'sonotype']
    print(f"  Total sonotype rows: {len(son)}")
    print(f"  Unique sonotype classes: {son['primary_label'].nunique()}")
    print(f"  Top sonotype classes:")
    for sp, ct in son['primary_label'].value_counts().head(10).items():
        n_tr = n_train.get(sp, 0)
        print(f"    {sp:<10}  n_tr={n_tr:>4d}  rows={ct:>6d}")

    # === 4. AXIS-2-AWARE FILTER DESIGN ===
    print(f"\n=== FILTER DESIGN — multi-tier criteria ===")
    # Tier A (common, n_train ≥ 20): keep all rows with v33 OR perch ≥ moderate
    # Tier B (rare, 5 ≤ n_train < 20): need agreement (v33 + perch)
    # Tier C (ultra-rare, n_train < 5): strictest multi-criteria + sonotype source filter

    # Filter rules
    # Source 'v2' = v33 teacher confidence
    # Source 'sonotype' = cosine-similarity-based, no v33 score

    keep_mask = pd.Series(False, index=v7.index)

    # Tier A: common class (n_train ≥ 20)
    tier_a = v7["n_train"] >= 20
    # accept if v33 ≥ 0.6 OR perch ≥ 0.3 (loose)
    keep_a = tier_a & ((v7["v33_score"] >= 0.6) | (v7["perch_score"] >= 0.3))
    keep_mask |= keep_a
    print(f"  Tier A (n_train ≥ 20):       {tier_a.sum():>7d} rows → keep {keep_a.sum():>7d} ({100*keep_a.sum()/max(tier_a.sum(),1):.1f}%)")

    # Tier B: rare class (5 ≤ n_train < 20)
    tier_b = (v7["n_train"] >= 5) & (v7["n_train"] < 20)
    # accept if v33 ≥ 0.7 AND (perch ≥ 0.3 OR emb_sim ≥ 0.4)
    keep_b = tier_b & (v7["v33_score"] >= 0.7) & ((v7["perch_score"] >= 0.3) | (v7["emb_sim"] >= 0.4))
    keep_mask |= keep_b
    print(f"  Tier B (5 ≤ n_train < 20):  {tier_b.sum():>7d} rows → keep {keep_b.sum():>7d} ({100*keep_b.sum()/max(tier_b.sum(),1):.1f}%)")

    # Tier C: ultra-rare class (n_train < 5)
    # For 'v2' source: require v33 ≥ 0.8 AND perch ≥ 0.3 AND emb_sim ≥ 0.5
    # For 'sonotype' source (cosine-based only): cos_sim ≥ 0.7
    tier_c = v7["n_train"] < 5
    keep_c_v2 = tier_c & (v7["source"] == "v2") & (v7["v33_score"] >= 0.8) & (v7["perch_score"] >= 0.3) & (v7["emb_sim"] >= 0.5)
    keep_c_son = tier_c & (v7["source"] == "sonotype") & (v7["cos_sim"] >= 0.7)
    keep_c = keep_c_v2 | keep_c_son
    keep_mask |= keep_c
    print(f"  Tier C (n_train < 5):       {tier_c.sum():>7d} rows → keep {keep_c.sum():>7d} ({100*keep_c.sum()/max(tier_c.sum(),1):.1f}%)")
    print(f"    Tier C/v2:                                  {keep_c_v2.sum():>7d}")
    print(f"    Tier C/sonotype:                            {keep_c_son.sum():>7d}")

    print(f"\n  TOTAL: {len(v7):>7d} → keep {keep_mask.sum():>7d} ({100*keep_mask.sum()/len(v7):.1f}%)")

    # Produce v10
    v10 = v7[keep_mask].copy().reset_index(drop=True)

    # Add a column indicating the filter tier
    v10["tier"] = "?"
    v10.loc[v10["n_train"] >= 20, "tier"] = "A_common"
    v10.loc[(v10["n_train"] >= 5) & (v10["n_train"] < 20), "tier"] = "B_rare"
    v10.loc[(v10["n_train"] < 5) & (v10["source"] == "v2"), "tier"] = "C_ultra_v2"
    v10.loc[(v10["n_train"] < 5) & (v10["source"] == "sonotype"), "tier"] = "C_ultra_sonotype"

    print(f"\n=== v10 per-tier counts ===")
    print(v10["tier"].value_counts())

    # Per-class
    v10_by_cls = v10.groupby("primary_label").size().reset_index(name="rows_v10")
    v10_by_cls["n_train"] = v10_by_cls["primary_label"].map(n_train).fillna(0).astype(int)
    v10_by_cls["taxon"] = v10_by_cls["primary_label"].map(sp_to_tax).fillna("?")
    v7_by_cls = v7.groupby("primary_label").size().reset_index(name="rows_v7")
    cmp = v10_by_cls.merge(v7_by_cls, on="primary_label", how="left")
    cmp["drop_pct"] = 100 * (1 - cmp["rows_v10"] / cmp["rows_v7"])
    cmp = cmp.sort_values("drop_pct", ascending=False)

    print(f"\n=== Top 20 classes most aggressively filtered (% rows dropped) ===")
    print(f"  {'species':<10}  {'tx':<8}  {'n_tr':>5}  {'v7_rows':>8}  {'v10_rows':>8}  {'drop%':>6}")
    for _, r in cmp.head(20).iterrows():
        print(f"  {r['primary_label']:<10}  {r['taxon']:<8}  {r['n_train']:>5d}  {r['rows_v7']:>8d}  {r['rows_v10']:>8d}  {r['drop_pct']:>5.1f}%")

    # Save
    v10.to_csv(OUT_V10, index=False)
    print(f"\nSaved v10: {OUT_V10}  ({len(v10)} rows)")


if __name__ == "__main__":
    main()

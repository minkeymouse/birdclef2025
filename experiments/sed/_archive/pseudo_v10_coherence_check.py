"""Pseudo v10 — window-level coherence check (heuristic quality check).

For each (filename, start, end) window, what's the multi-label pattern?
- Common + ultra-rare cooccur: legit (multi-species in same time window)
- All-Insecta sonotypes only: likely sonotype noise (Insecta calls don't overlap)
- All-Amphibia + common Aves: legit (different taxa, can cooccur)

Filter idea (Tier D, optional): drop sonotype labels that appear ONLY with
other sonotypes in the same window (likely over-prediction artifact). Keep
sonotypes that cooccur with at least one v2-source label (validated by v33).

Also: check 'sonotype' source distribution by site — are S06 (which produced
many sonotype rows) site-specific?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
V10 = ROOT / "data/birdclef-2026/pseudo_soundscapes_labels_v10.csv"
TRAIN_CSV = ROOT / "data/birdclef-2026/train.csv"


def main():
    print("Loading v10...")
    v10 = pd.read_csv(V10)
    train = pd.read_csv(TRAIN_CSV)
    n_train = train.groupby("primary_label").size().to_dict()
    print(f"v10: {v10.shape}")

    # Add site
    v10["site"] = v10["filename"].str.extract(r"(S\d+)")[0]
    v10["n_train"] = v10["primary_label"].map(n_train).fillna(0).astype(int)

    # === 1. Site distribution per source ===
    print("\n=== Sonotype source per-site distribution ===")
    son = v10[v10["source"] == "sonotype"]
    site_son = son.groupby("site").size().sort_values(ascending=False)
    print(f"  Total sonotype rows in v10: {len(son)}")
    print(f"  {'site':<6}  {'rows':>6}  {'%':>6}")
    for s, ct in site_son.items():
        print(f"  {s:<5}  {ct:>6d}  {100*ct/len(son):>5.1f}%")

    # === 2. Per-window multi-label pattern ===
    print("\n=== Per-window co-occurrence ===")
    # Group by (filename, start, end) to get all labels for each window
    window_groups = v10.groupby(["filename", "start", "end"])
    print(f"  Unique windows in v10: {len(window_groups)}")

    # For each window, get the set of labels and their sources
    window_stats = []
    for (fn, start, end), grp in window_groups:
        labels = grp["primary_label"].tolist()
        sources = grp["source"].tolist()
        n_son = sum(1 for s in sources if s == "sonotype")
        n_v2 = sum(1 for s in sources if s == "v2")
        n_total = len(labels)
        window_stats.append({
            "filename": fn, "start": start, "end": end,
            "n_total": n_total, "n_son": n_son, "n_v2": n_v2,
            "labels": labels,
        })
    ws = pd.DataFrame(window_stats)

    print(f"  Multi-label windows (≥2 labels): {(ws.n_total >= 2).sum()} ({100*(ws.n_total >= 2).sum()/len(ws):.1f}%)")
    print(f"  Sonotype-only windows (no v2 labels at all): {((ws.n_son >= 1) & (ws.n_v2 == 0)).sum()}")
    print(f"  v2-only windows: {((ws.n_v2 >= 1) & (ws.n_son == 0)).sum()}")
    print(f"  Mixed (v2 + sonotype): {((ws.n_v2 >= 1) & (ws.n_son >= 1)).sum()}")

    # === 3. Multi-sonotype windows: same window has multiple sonotype labels ===
    print("\n=== Multi-sonotype windows (≥2 sonotype labels in same window) ===")
    multi_son = ws[ws.n_son >= 2]
    print(f"  Windows with ≥2 sonotype labels: {len(multi_son)}")
    if len(multi_son) > 0:
        print("\n  Sample 5:")
        sample = multi_son.sample(min(5, len(multi_son)), random_state=42)
        for _, r in sample.iterrows():
            labs = r['labels']
            n_son_in = sum(1 for sp in labs if sp.startswith("47158son"))
            print(f"    {r['filename']:<40} t={r['start']:>3d}-{r['end']:>3d}  labels: {labs[:4]}{'...' if len(labs)>4 else ''}  (sonotypes: {n_son_in}/{len(labs)})")

    # === 4. Co-occurrence matrix: sonotype labels — which pairs cooccur most ===
    print("\n=== Sonotype co-occurrence in same window ===")
    son_co = {}
    for _, r in ws.iterrows():
        son_labs = [l for l in r['labels'] if l.startswith("47158son")]
        if len(son_labs) >= 2:
            for i, a in enumerate(son_labs):
                for b in son_labs[i+1:]:
                    pair = tuple(sorted([a, b]))
                    son_co[pair] = son_co.get(pair, 0) + 1
    if son_co:
        top = sorted(son_co.items(), key=lambda x: -x[1])[:10]
        print(f"  Top 10 sonotype co-occurring pairs:")
        for (a, b), ct in top:
            print(f"    {a:<11} + {b:<11}: {ct} windows")

    # === 5. Tier D filter idea — keep sonotype only if cooccurs with v2 label
    print("\n=== Tier D filter (cross-validation): sonotype kept only if window also has v2 label ===")
    # Build (filename, start, end) → has_v2 set
    has_v2 = set()
    v2_rows = v10[v10["source"] == "v2"]
    for _, r in v2_rows.iterrows():
        has_v2.add((r["filename"], r["start"], r["end"]))
    print(f"  Unique windows with v2 label: {len(has_v2)}")

    son_in_v2_window = []
    for _, r in son.iterrows():
        key = (r["filename"], r["start"], r["end"])
        son_in_v2_window.append(key in has_v2)
    son_in_v2_window = np.array(son_in_v2_window)
    print(f"  Sonotype rows where window also has v2 label: {son_in_v2_window.sum()} / {len(son)}  ({100*son_in_v2_window.sum()/len(son):.1f}%)")
    print(f"  → Tier D 'strict' would drop {(~son_in_v2_window).sum()} sonotype rows (no v2 cross-validation)")

    # === 6. Per-class final v10 distribution
    print("\n=== Final v10 per-class distribution (top 30) ===")
    by_cls = v10.groupby("primary_label").size().reset_index(name="rows")
    by_cls["n_train"] = by_cls["primary_label"].map(n_train).fillna(0).astype(int)
    by_cls = by_cls.sort_values("rows", ascending=False)
    print(f"  {'species':<10}  {'n_tr':>5}  {'v10_rows':>8}")
    for _, r in by_cls.head(20).iterrows():
        print(f"  {r['primary_label']:<10}  {r['n_train']:>5d}  {r['rows']:>8d}")

    # ratio common vs ultra-rare in v10
    print(f"\n  Common (n_train≥20): {(by_cls['n_train']>=20).sum()} classes, {by_cls[by_cls['n_train']>=20]['rows'].sum()} rows")
    print(f"  Rare (5≤n_train<20): {((by_cls['n_train']>=5) & (by_cls['n_train']<20)).sum()} classes, {by_cls[(by_cls['n_train']>=5) & (by_cls['n_train']<20)]['rows'].sum()} rows")
    print(f"  Ultra-rare (n_train<5): {(by_cls['n_train']<5).sum()} classes, {by_cls[by_cls['n_train']<5]['rows'].sum()} rows")


if __name__ == "__main__":
    main()

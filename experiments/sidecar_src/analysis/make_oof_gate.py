import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    if np.nanstd(p) <= 1e-12:
        return np.nan
    return float(roc_auc_score(y, p))


def shrink_auc(auc, n_pos, tau):
    if not np.isfinite(auc):
        return np.nan
    rho = float(n_pos) / (float(n_pos) + float(tau))
    return float(0.5 + rho * (float(auc) - 0.5))


def frequency_bucket(n_pos):
    n_pos = int(n_pos)
    if n_pos < 3:
        return "nlt3"
    if n_pos < 6:
        return "n3_5"
    if n_pos < 12:
        return "n6_11"
    return "n12p"


def read_labels(oof, sample_path):
    if "labels" in oof.files:
        return [str(x) for x in oof["labels"]]
    if "class_names" in oof.files:
        return [str(x) for x in oof["class_names"]]
    sample = pd.read_csv(sample_path)
    return sample.columns[1:].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True, help="OOF npz with y_true and pred_oof")
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tau-class", type=float, default=20.0)
    parser.add_argument("--tau-group", type=float, default=30.0)
    parser.add_argument("--min-pos-class", type=int, default=3)
    parser.add_argument("--min-pred-std", type=float, default=1e-4)
    parser.add_argument("--class-threshold", type=float, default=0.545)
    parser.add_argument("--group-threshold", type=float, default=0.535)
    parser.add_argument("--class-scale", type=float, default=0.075)
    parser.add_argument("--group-scale", type=float, default=0.075)
    parser.add_argument("--min-factor", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=0.030)
    args = parser.parse_args()

    oof = np.load(args.oof, allow_pickle=True)
    if "y_true" not in oof.files:
        raise KeyError(f"y_true missing from {args.oof}; available={oof.files}")
    pred_key = "pred_oof" if "pred_oof" in oof.files else "pred"
    if pred_key not in oof.files:
        raise KeyError(f"pred_oof/pred missing from {args.oof}; available={oof.files}")

    y = oof["y_true"].astype(np.float32)
    p = oof[pred_key].astype(np.float32)
    labels = read_labels(oof, args.sample)

    sample = pd.read_csv(args.sample)
    sample_labels = sample.columns[1:].astype(str).tolist()
    if labels != sample_labels:
        raise AssertionError("OOF label order differs from sample_submission")
    if y.shape != p.shape:
        raise AssertionError(f"y/p shape mismatch: y={y.shape}, p={p.shape}")
    if y.shape[1] != len(labels):
        raise AssertionError(f"label count mismatch: y={y.shape}, labels={len(labels)}")

    tax = pd.read_csv(args.taxonomy)
    tax["primary_label"] = tax["primary_label"].astype(str)
    if "class_name" in tax.columns:
        class_name_map = dict(zip(tax["primary_label"], tax["class_name"].astype(str)))
    else:
        class_name_map = {}

    rows = []
    for j, label in enumerate(labels):
        yy = y[:, j]
        pp = p[:, j]
        n_pos = int(yy.sum())
        auc = safe_auc(yy, pp)
        auc_shrunk = shrink_auc(auc, n_pos, args.tau_class)
        pred_std = float(np.nanstd(pp))
        class_name = str(class_name_map.get(label, "unknown"))
        freq = frequency_bucket(n_pos)
        rows.append(
            {
                "class_idx": j,
                "primary_label": label,
                "class_name": class_name,
                "n_pos": n_pos,
                "n_neg": int(len(yy) - n_pos),
                "auc": auc,
                "auc_shrunk": auc_shrunk,
                "pred_std": pred_std,
                "freq_bucket": freq,
                "group": f"{class_name}_{freq}",
            }
        )

    df = pd.DataFrame(rows)
    group_rows = []
    for group, g in df.groupby("group", sort=True):
        valid = g[np.isfinite(g["auc"])].copy()
        group_pos = int(g["n_pos"].sum())
        if len(valid):
            weights = np.sqrt(valid["n_pos"].clip(lower=1).to_numpy(dtype=np.float64))
            group_auc = float(np.average(valid["auc"].to_numpy(dtype=np.float64), weights=weights))
            group_auc_shrunk = shrink_auc(group_auc, int(valid["n_pos"].sum()), args.tau_group)
        else:
            group_auc = np.nan
            group_auc_shrunk = np.nan
        group_rows.append(
            {
                "group": group,
                "group_auc": group_auc,
                "group_auc_shrunk": group_auc_shrunk,
                "group_pos": group_pos,
                "group_classes": int(len(g)),
            }
        )

    df = df.merge(pd.DataFrame(group_rows), on="group", how="left")

    gate_weights = []
    reasons = []
    for _, row in df.iterrows():
        n_pos = int(row["n_pos"])
        auc_s = float(row["auc_shrunk"]) if np.isfinite(row["auc_shrunk"]) else np.nan
        group_s = (
            float(row["group_auc_shrunk"]) if np.isfinite(row["group_auc_shrunk"]) else np.nan
        )
        pred_std = float(row["pred_std"])

        weight = 0.0
        reason = "off"
        if n_pos < args.min_pos_class:
            reason = "too_few_positive"
        elif pred_std < args.min_pred_std:
            reason = "constant_prediction"
        elif not np.isfinite(auc_s):
            reason = "auc_nan"
        elif not np.isfinite(group_s):
            reason = "group_auc_nan"
        elif group_s < args.group_threshold:
            reason = "weak_group"
        elif auc_s < args.class_threshold:
            reason = "weak_class"
        else:
            class_factor = np.clip(
                (auc_s - args.class_threshold) / max(args.class_scale, 1e-12), 0.0, 1.0
            )
            group_factor = np.clip(
                (group_s - args.group_threshold) / max(args.group_scale, 1e-12), 0.0, 1.0
            )
            factor = float(class_factor * group_factor)
            if factor <= args.min_factor:
                reason = "factor_too_low"
            else:
                weight = float(args.max_weight * factor)
                reason = "enabled"
        gate_weights.append(weight)
        reasons.append(reason)

    df["gate_weight"] = gate_weights
    df["gate_reason"] = reasons
    df["gate_enabled"] = df["gate_weight"] > 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    enabled = df[df["gate_enabled"]]
    summary = {
        "classes": int(len(df)),
        "enabled_classes": int(len(enabled)),
        "mean_gate_weight": float(df["gate_weight"].mean()),
        "max_gate_weight": float(df["gate_weight"].max()),
        "enabled_by_class_name": enabled.groupby("class_name").size().to_dict(),
        "out": str(out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

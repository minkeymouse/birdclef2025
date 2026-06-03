import argparse

import numpy as np
import pandas as pd


def read_sub(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"row_id missing: {path}")
    df["row_id"] = df["row_id"].astype(str)
    df = df.set_index("row_id")
    if not df.index.is_unique:
        raise ValueError(f"duplicate row_id: {path}")
    return df


def rank_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=0, method="average", pct=True).astype(np.float32)


def topk_mask(arr: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), arr.shape[1])
    idx = np.argpartition(arr, -k, axis=1)[:, -k:]
    mask = np.zeros_like(arr, dtype=bool)
    rows = np.arange(arr.shape[0])[:, None]
    mask[rows, idx] = True
    return mask


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(max(int(k), 1), a.shape[1])
    ai = np.argpartition(a, -k, axis=1)[:, -k:]
    bi = np.argpartition(b, -k, axis=1)[:, -k:]
    return float(np.mean([len(set(x).intersection(set(y))) / k for x, y in zip(ai, bi)]))


def load_gate_weights(path: str | None, columns: list[str], max_weight: float) -> np.ndarray | None:
    if not path:
        return None
    gate = pd.read_csv(path)
    if "primary_label" not in gate.columns or "gate_weight" not in gate.columns:
        raise ValueError("gate CSV must contain primary_label and gate_weight")
    gate["primary_label"] = gate["primary_label"].astype(str)
    weight_map = dict(zip(gate["primary_label"], gate["gate_weight"].astype(float)))
    weights = np.array([float(weight_map.get(str(c), 0.0)) for c in columns], dtype=np.float32)
    return np.clip(weights, 0.0, float(max_weight))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--w-cap", type=float, default=0.05)
    parser.add_argument("--d-budget", type=float, default=0.006)
    parser.add_argument("--anchor-topk", type=int, default=48)
    parser.add_argument("--side-topk", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.55)
    parser.add_argument("--min-top3", type=float, default=0.75)
    parser.add_argument("--min-top10", type=float, default=0.80)
    parser.add_argument("--gate-csv", default=None)
    parser.add_argument("--gate-max-weight", type=float, default=0.03)
    args = parser.parse_args()

    anchor = read_sub(args.anchor)
    sidecar = read_sub(args.sidecar)
    sidecar = sidecar.loc[anchor.index, anchor.columns]

    a = rank_cols(anchor).to_numpy(np.float32)
    s = rank_cols(sidecar).to_numpy(np.float32)
    mask = topk_mask(a, args.anchor_topk) | (topk_mask(s, args.side_topk) & (a >= args.tau))
    e = float(np.mean(mask * np.abs(s - a)))
    gate_weights = load_gate_weights(args.gate_csv, anchor.columns.tolist(), args.gate_max_weight)
    b = a.copy()
    if gate_weights is None:
        w = 0.0 if e <= 1e-12 else min(args.w_cap, args.d_budget / e)
        if w > 0:
            b[mask] = a[mask] + w * (s[mask] - a[mask])
        mode = "scalar"
        nonzero_classes = a.shape[1]
    else:
        ww = np.broadcast_to(gate_weights[None, :], a.shape)
        if mask.any() and gate_weights.max() > 0:
            b[mask] = a[mask] + ww[mask] * (s[mask] - a[mask])
        d0 = float(np.mean(np.abs(b - a)))
        if d0 > args.d_budget and d0 > 1e-12:
            scale = args.d_budget / d0
            gate_weights = gate_weights * scale
            ww = np.broadcast_to(gate_weights[None, :], a.shape)
            b = a.copy()
            b[mask] = a[mask] + ww[mask] * (s[mask] - a[mask])
        w = float(gate_weights.mean())
        mode = "oof_gated"
        nonzero_classes = int((gate_weights > 0).sum())
    top3 = topk_overlap(a, b, 3)
    top10 = topk_overlap(a, b, 10)
    ok = top3 >= args.min_top3 and top10 >= args.min_top10
    print(
        {
            "mode": mode,
            "w": w,
            "nonzero_classes": nonzero_classes,
            "E": e,
            "D": float(np.mean(np.abs(b - a))),
            "active_fraction": float(mask.mean()),
            "top3_overlap": top3,
            "top10_overlap": top10,
            "ok": ok,
        }
    )
    out_arr = b if ok else a
    out = pd.DataFrame(out_arr, index=anchor.index, columns=anchor.columns).reset_index()
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

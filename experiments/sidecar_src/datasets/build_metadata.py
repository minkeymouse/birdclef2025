import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def hms_to_seconds(value) -> int:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        raise ValueError(f"Bad time value: {value}")
    if ":" not in s:
        return int(round(float(s)))
    return int(pd.to_timedelta(s).total_seconds())


def parse_labels(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip() and x.strip().lower() != "nan"]


def build_label_lookup(label_df: pd.DataFrame, label_to_idx: dict[str, int]) -> dict[tuple[str, int], list[int]]:
    lookup: dict[tuple[str, int], list[int]] = {}
    for _, row in label_df.iterrows():
        filename = str(row["filename"])
        end_sec = hms_to_seconds(row["end"])
        labs = []
        for lab in parse_labels(row["primary_label"]):
            if lab in label_to_idx:
                labs.append(label_to_idx[lab])
        if labs:
            lookup.setdefault((filename, end_sec), []).extend(labs)
    return lookup


def fully_labeled_files(label_df: pd.DataFrame) -> set[str]:
    tmp = label_df.copy()
    tmp["end_sec"] = tmp["end"].map(hms_to_seconds)
    expected = set(range(5, 61, 5))
    full = []
    for filename, group in tmp.groupby("filename", sort=False):
        ends = set(group["end_sec"].astype(int).tolist())
        if ends == expected:
            full.append(filename)
    return set(full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-seconds", type=float, default=10.0)
    parser.add_argument("--target-seconds", type=float, default=5.0)
    parser.add_argument("--fully-labeled-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    labels = [str(c) for c in sample.columns[1:]]
    label_to_idx = {c: i for i, c in enumerate(labels)}
    labels_df = pd.DataFrame({"class_idx": np.arange(len(labels)), "primary_label": labels})
    labels_df.to_csv(out_dir / "labels.csv", index=False)

    folds = pd.read_csv(args.folds)
    folds["filename"] = folds["filename"].astype(str)
    label_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    label_df["filename"] = label_df["filename"].astype(str)
    full_files = fully_labeled_files(label_df)
    if args.fully_labeled_only:
        before = len(folds)
        folds = folds[folds["filename"].isin(full_files)].reset_index(drop=True)
        print(f"Fully labeled metadata files: {len(folds)} / {before}")
    if folds.empty:
        raise RuntimeError("No files left after fully-labeled filtering")

    label_lookup = build_label_lookup(label_df, label_to_idx)

    half_extra = max(0.0, (float(args.context_seconds) - float(args.target_seconds)) / 2.0)
    rows = []
    y = np.zeros((len(folds) * 12, len(labels)), dtype=np.uint8)
    write = 0
    missing_labeled_windows = 0
    for _, file_row in folds.iterrows():
        filename = str(file_row["filename"])
        file_id = str(file_row["file_id"])
        is_full = filename in full_files
        for window_idx in range(12):
            target_start = window_idx * float(args.target_seconds)
            target_end = target_start + float(args.target_seconds)
            end_int = int(round(target_end))
            row_id = f"{file_id}_{end_int}"
            labs = label_lookup.get((filename, end_int), [])
            if labs:
                y[write, labs] = 1
            elif is_full:
                pass
            else:
                missing_labeled_windows += 1
            rows.append(
                {
                    "row_id": row_id,
                    "file_id": file_id,
                    "filename": filename,
                    "path": str(data_dir / "train_soundscapes" / filename),
                    "window_idx": window_idx,
                    "target_start_sec": target_start,
                    "target_end_sec": target_end,
                    "context_start_sec": target_start - half_extra,
                    "context_end_sec": target_end + half_extra,
                    "fold": int(file_row["fold"]),
                    "site": file_row.get("site", ""),
                    "hour": int(file_row.get("hour", -1)),
                    "num_positive": int(len(labs)),
                    "is_fully_labeled": bool(is_full),
                }
            )
            write += 1

    meta = pd.DataFrame(rows)
    assert len(meta) == y.shape[0]
    assert meta["row_id"].is_unique

    meta_path = out_dir / "soundscape_windows.parquet"
    y_path = out_dir / "y_soundscape.npy"
    meta.to_parquet(meta_path, index=False)
    np.save(y_path, y)
    print(f"Wrote {meta_path}: rows={len(meta)}, positive_windows={(y.sum(axis=1) > 0).sum()}")
    print(f"Wrote {y_path}: shape={y.shape}, positives={int(y.sum())}")
    if missing_labeled_windows:
        print(f"Warning: missing labeled windows treated as zero labels: {missing_labeled_windows}")


if __name__ == "__main__":
    main()

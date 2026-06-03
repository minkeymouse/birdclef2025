import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def parse_label_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip().strip("'\"") for x in text.replace(",", ";").split(";") if x.strip().strip("'\"[]")]


def safe_duration_sec(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return 0.0


def segment_start_times(duration: float, context_seconds: float, segments_per_file: int) -> list[float]:
    if duration <= context_seconds:
        return [0.0]
    if segments_per_file <= 1:
        return [max(0.0, (duration - context_seconds) / 2.0)]
    max_start = max(0.0, duration - context_seconds)
    starts = np.linspace(0.0, max_start, segments_per_file, dtype=np.float32)
    starts = np.unique(np.round(starts, 3))
    return starts.astype(float).tolist()


def select_balanced_audio(
    train: pd.DataFrame,
    max_per_class: int,
    min_rating: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    train = train.copy()
    train["_rand"] = rng.random(len(train))
    if "rating" in train.columns:
        train["rating"] = pd.to_numeric(train["rating"], errors="coerce").fillna(0.0)
        train = train[train["rating"] >= float(min_rating)]
        sort_cols = ["rating", "_rand"]
        ascending = [False, True]
    else:
        sort_cols = ["_rand"]
        ascending = [True]
    for _, group in train.groupby("primary_label", sort=False):
        group = group.sort_values(sort_cols, ascending=ascending)
        if max_per_class > 0:
            group = group.head(max_per_class)
        rows.append(group)
    if not rows:
        raise RuntimeError("No train_audio rows selected")
    out = pd.concat(rows, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.drop(columns=["_rand"], errors="ignore")


def assign_label_round_robin_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    fold = np.full(len(df), -1, dtype=np.int16)
    rng = np.random.default_rng(seed)
    for _, idx in df.groupby("primary_label", sort=False).groups.items():
        idx = np.array(list(idx), dtype=np.int64)
        rng.shuffle(idx)
        for j, row_idx in enumerate(idx):
            fold[row_idx] = j % n_folds
    if (fold < 0).any():
        raise RuntimeError("Internal fold assignment error")
    return pd.Series(fold, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7177)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--context-seconds", type=float, default=10.0)
    parser.add_argument("--segments-per-file", type=int, default=1)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--secondary-weight", type=float, default=0.5)
    parser.add_argument("--negative-mask-weight", type=float, default=0.15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    labels = [str(c) for c in sample.columns[1:]]
    label_to_idx = {c: i for i, c in enumerate(labels)}
    train = pd.read_csv(data_dir / "train.csv")
    train["primary_label"] = train["primary_label"].astype(str)
    train["filename"] = train["filename"].astype(str)
    train = train[train["primary_label"].isin(label_to_idx)].reset_index(drop=True)
    train["path"] = [str(data_dir / "train_audio" / fn) for fn in train["filename"]]
    train = train[train["path"].map(lambda x: Path(x).exists())].reset_index(drop=True)
    if train.empty:
        raise RuntimeError("No train_audio files found after filtering")

    selected = select_balanced_audio(
        train,
        max_per_class=int(args.max_per_class),
        min_rating=float(args.min_rating),
        seed=int(args.seed),
    )
    selected["fold"] = assign_label_round_robin_folds(selected, int(args.n_folds), int(args.seed))

    rows = []
    targets = []
    masks = []
    context_seconds = float(args.context_seconds)
    for src_i, row in selected.reset_index(drop=True).iterrows():
        path = Path(str(row["path"]))
        duration = safe_duration_sec(path)
        starts = segment_start_times(duration, context_seconds, int(args.segments_per_file))
        primary = str(row["primary_label"])
        secondary = [lab for lab in parse_label_list(row.get("secondary_labels", "[]")) if lab in label_to_idx]
        for seg_i, start in enumerate(starts):
            target = np.zeros(len(labels), dtype=np.float32)
            mask = np.full(len(labels), float(args.negative_mask_weight), dtype=np.float32)
            target[label_to_idx[primary]] = float(args.primary_weight)
            mask[label_to_idx[primary]] = 1.0
            for lab in secondary:
                target[label_to_idx[lab]] = max(target[label_to_idx[lab]], float(args.secondary_weight))
                mask[label_to_idx[lab]] = 1.0
            file_stem = Path(str(row["filename"])).with_suffix("").as_posix().replace("/", "__")
            rows.append(
                {
                    "row_id": f"audio_{file_stem}_{seg_i}",
                    "file_id": file_stem,
                    "filename": str(row["filename"]),
                    "path": str(path),
                    "window_idx": int(seg_i),
                    "target_start_sec": float(start),
                    "target_end_sec": float(start + context_seconds),
                    "context_start_sec": float(start),
                    "context_end_sec": float(start + context_seconds),
                    "fold": int(row["fold"]),
                    "site": "",
                    "hour": -1,
                    "num_positive": int((target > 0).sum()),
                    "source": "train_audio",
                    "duration_sec": float(duration),
                    "primary_label": primary,
                    "secondary_count": int(len(secondary)),
                }
            )
            targets.append(target)
            masks.append(mask)

    meta = pd.DataFrame(rows)
    y = np.stack(targets, axis=0).astype(np.float32)
    target_mask = np.stack(masks, axis=0).astype(np.float32)
    assert meta["row_id"].is_unique
    assert len(meta) == y.shape[0] == target_mask.shape[0]

    meta_path = out_dir / "train_audio_windows.parquet"
    y_path = out_dir / "y_train_audio.npy"
    mask_path = out_dir / "mask_train_audio.npy"
    meta.to_parquet(meta_path, index=False)
    np.save(y_path, y)
    np.save(mask_path, target_mask)
    print(f"Wrote {meta_path}: rows={len(meta)}, files={selected['filename'].nunique()}")
    print(f"Wrote {y_path}: shape={y.shape}, positive_entries={int((y > 0).sum())}")
    print(f"Wrote {mask_path}: shape={target_mask.shape}, mean_mask={float(target_mask.mean()):.4f}")
    print(meta.groupby("fold").agg(rows=("row_id", "count"), positives=("num_positive", "sum")))


if __name__ == "__main__":
    main()

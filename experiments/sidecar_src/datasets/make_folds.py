import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_site_hour(stem: str) -> tuple[str, int]:
    m_site = re.search(r"_(S\d{2})_", stem)
    site = m_site.group(1) if m_site else "S00"
    m_time = re.search(r"_(\d{8})_(\d{6})$", stem)
    hour = int(m_time.group(2)[:2]) if m_time else -1
    return site, hour


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


def build_file_label_matrix(label_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    rows = []
    for filename, g in label_df.groupby("filename", sort=False):
        vec = np.zeros(len(class_names), dtype=np.uint8)
        for labs in g["primary_label"]:
            for lab in parse_labels(labs):
                if lab in label_to_idx:
                    vec[label_to_idx[lab]] = 1
        rows.append({"filename": filename, "label_vec": vec, "num_positive": int(vec.sum())})
    return pd.DataFrame(rows)


def fully_labeled_files(label_df: pd.DataFrame) -> set[str]:
    expected = set(range(5, 61, 5))
    full = []
    for filename, group in label_df.groupby("filename", sort=False):
        ends = set(group["end_sec"].astype(int).tolist())
        if ends == expected:
            full.append(filename)
    return set(full)


def multilabel_greedy_folds(file_df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = file_df.copy()
    label_mat = np.stack(df["label_vec"].to_numpy())
    class_totals = label_mat.sum(axis=0).astype(np.float64)
    class_weight = 1.0 / np.sqrt(class_totals + 1.0)
    rarity_score = (label_mat * class_weight[None, :]).sum(axis=1)
    df["_rarity"] = rarity_score
    df["_jitter"] = rng.random(len(df))
    df = df.sort_values(["_rarity", "num_positive", "_jitter"], ascending=[False, False, True]).reset_index(drop=True)

    fold_counts = np.zeros((n_folds, label_mat.shape[1]), dtype=np.float64)
    fold_n = np.zeros(n_folds, dtype=np.float64)
    target_per_fold = class_totals / float(n_folds)
    target_size = len(df) / float(n_folds)
    max_size = int(np.ceil(target_size))
    folds = []
    for i, (_, row) in enumerate(df.iterrows()):
        y = row["label_vec"].astype(np.float64)
        if i < n_folds:
            f = i
            folds.append(f)
            fold_counts[f] += y
            fold_n[f] += 1.0
            continue
        scores = []
        for f in range(n_folds):
            if fold_n[f] >= max_size and (fold_n < max_size).any():
                scores.append(float("inf"))
                continue
            trial = fold_counts[f] + y
            class_cost = (((trial - target_per_fold) ** 2) * class_weight).sum()
            size_cost = 10.0 * ((fold_n[f] + 1.0 - target_size) ** 2) / max(target_size, 1.0)
            scores.append(class_cost + size_cost)
        f = int(np.argmin(scores))
        folds.append(f)
        fold_counts[f] += y
        fold_n[f] += 1.0
    df["fold"] = folds
    return df.drop(columns=["_rarity", "_jitter", "label_vec"]).sort_values("file_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7177)
    parser.add_argument("--fully-labeled-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    soundscape_dir = data_dir / "train_soundscapes"
    label_path = data_dir / "train_soundscapes_labels.csv"
    sample_path = data_dir / "sample_submission.csv"
    if not soundscape_dir.exists():
        raise FileNotFoundError(soundscape_dir)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    class_names = [str(c) for c in pd.read_csv(sample_path, nrows=1).columns[1:]]
    files = sorted(soundscape_dir.glob("*.ogg"))
    labels = pd.read_csv(label_path)
    labels["filename"] = labels["filename"].astype(str)
    labels["end_sec"] = labels["end"].map(hms_to_seconds)

    full_files = fully_labeled_files(labels)
    if args.fully_labeled_only:
        print(f"Using fully labeled files only: {len(full_files)} / {len(files)}")
    else:
        print("Using all train_soundscapes; missing label windows will be treated as zero labels")

    file_label_df = build_file_label_matrix(labels, class_names)
    file_label_map = file_label_df.set_index("filename")

    rows = []
    skipped = 0
    for path in files:
        if args.fully_labeled_only and path.name not in full_files:
            skipped += 1
            continue
        site, hour = parse_site_hour(path.stem)
        if path.name in file_label_map.index:
            num_positive = int(file_label_map.loc[path.name, "num_positive"])
            label_vec = file_label_map.loc[path.name, "label_vec"]
        else:
            num_positive = 0
            label_vec = np.zeros(len(class_names), dtype=np.uint8)
        rows.append(
            {
                "file_id": path.stem,
                "filename": path.name,
                "path": str(path),
                "site": site,
                "hour": hour,
                "num_positive": num_positive,
                "label_vec": label_vec,
                "is_fully_labeled": path.name in full_files,
            }
        )
    if not rows:
        raise RuntimeError("No soundscape files selected for folds")
    file_df = pd.DataFrame(rows)
    out = multilabel_greedy_folds(file_df, args.n_folds, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: files={len(out)}, skipped={skipped}, folds={args.n_folds}")
    print(out.groupby("fold").agg(files=("file_id", "count"), positives=("num_positive", "sum")))


if __name__ == "__main__":
    main()

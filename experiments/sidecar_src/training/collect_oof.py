import argparse
from pathlib import Path

import numpy as np

from sidecar_src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["output"]["dir"])
    rows, file_ids, filenames, window_idxs, y_true, pred, fold_arr = [], [], [], [], [], [], []
    labels = None
    for f in args.folds:
        path = out_root / f"fold{f}" / f"fold{f}_valid_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path, allow_pickle=True)
        rows.append(arr["row_id"].astype(str))
        file_ids.append(arr["file_id"].astype(str))
        filenames.append(arr["filename"].astype(str))
        window_idxs.append(arr["window_idx"].astype(np.int16))
        y_true.append(arr["y_true"].astype(np.uint8))
        pred.append(arr["pred"].astype(np.float32))
        fold_arr.append(arr["fold"].astype(np.int16))
        if labels is None:
            labels = arr["labels"].astype(str)

    row_id = np.concatenate(rows)
    order = np.argsort(row_id)
    out_path = Path(args.out) if args.out else out_root / "oof_predictions.npz"
    np.savez_compressed(
        out_path,
        row_id=row_id[order],
        file_id=np.concatenate(file_ids)[order],
        filename=np.concatenate(filenames)[order],
        window_idx=np.concatenate(window_idxs)[order],
        y_true=np.concatenate(y_true, axis=0)[order],
        pred_oof=np.concatenate(pred, axis=0)[order],
        fold=np.concatenate(fold_arr)[order],
        labels=labels,
    )
    print(f"Wrote {out_path}: rows={len(row_id)}")


if __name__ == "__main__":
    main()

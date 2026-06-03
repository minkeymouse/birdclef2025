import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

THIS_FILE = Path(__file__).resolve()
for parent in [THIS_FILE.parent, *THIS_FILE.parents]:
    if (parent / "sidecar_src").exists():
        sys.path.insert(0, str(parent))
        break

from sidecar_src.models.convnext_sed import build_model
from sidecar_src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default="submission_sidecar.csv")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run-files", type=int, default=1)
    return parser.parse_args()


def build_test_meta(paths: list[Path], cfg: dict) -> pd.DataFrame:
    target_seconds = float(cfg["audio"].get("target_seconds", 5.0))
    context_seconds = float(cfg["audio"].get("context_seconds", 10.0))
    half_extra = max(0.0, (context_seconds - target_seconds) / 2.0)
    rows = []
    for path in paths:
        stem = path.stem
        for window_idx in range(12):
            target_start = window_idx * target_seconds
            target_end = target_start + target_seconds
            rows.append(
                {
                    "row_id": f"{stem}_{int(round(target_end))}",
                    "file_id": stem,
                    "filename": path.name,
                    "path": str(path),
                    "window_idx": window_idx,
                    "target_start_sec": target_start,
                    "target_end_sec": target_end,
                    "context_start_sec": target_start - half_extra,
                    "context_end_sec": target_end + half_extra,
                }
            )
    return pd.DataFrame(rows)


def find_checkpoints(checkpoint_dir: Path, folds: list[str] | None) -> list[Path]:
    if folds:
        hits = []
        missing = []
        for f in folds:
            found = None
            for name in [f"fold{f}.pt", f"fold{f}/best.pt"]:
                p = checkpoint_dir / name
                if p.exists():
                    found = p
                    break
            if found is None:
                missing.append(str(f))
            else:
                hits.append(found)
        if missing:
            raise FileNotFoundError(f"Requested folds not found in {checkpoint_dir}: {missing}")
        return hits
    hits = sorted(checkpoint_dir.glob("fold*.pt"))
    if not hits:
        hits = sorted(checkpoint_dir.glob("fold*/best.pt"))
    return hits


def make_file_features(path: Path, cfg: dict) -> tuple[list[str], np.ndarray]:
    from sidecar_src.utils.audio import load_audio, slice_audio_segment
    from sidecar_src.utils.features import make_feature

    sr = int(cfg["audio"]["sample_rate"])
    target_seconds = float(cfg["audio"].get("target_seconds", 5.0))
    context_seconds = float(cfg["audio"].get("context_seconds", 10.0))
    half_extra = max(0.0, (context_seconds - target_seconds) / 2.0)
    wav = load_audio(path, sr)
    row_ids = []
    images = []
    for window_idx in range(12):
        target_start = window_idx * target_seconds
        target_end = target_start + target_seconds
        seg = slice_audio_segment(wav, target_start - half_extra, target_end + half_extra, sr)
        images.append(make_feature(seg, cfg))
        row_ids.append(f"{path.stem}_{int(round(target_end))}")
    return row_ids, np.stack(images, axis=0).astype(np.float32)


def load_model(ckpt_path: Path, cfg: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg, pretrained=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_all(checkpoints: list[Path], paths: list[Path], cfg: dict, device: torch.device, batch_size: int):
    models = [load_model(p, cfg, device) for p in checkpoints]
    row_ids_all = []
    pred_all = []
    for path in tqdm(paths, desc="files"):
        row_ids, images = make_file_features(path, cfg)
        x_all = torch.from_numpy(images).float()
        pred_sum = np.zeros((len(row_ids), int(cfg["model"]["num_classes"])), dtype=np.float32)
        for model in models:
            chunks = []
            for start in range(0, len(x_all), batch_size):
                x = x_all[start : start + batch_size].to(device, non_blocking=True)
                chunks.append(torch.sigmoid(model(x)).cpu().numpy())
            pred_sum += np.concatenate(chunks, axis=0).astype(np.float32)
        row_ids_all.extend(row_ids)
        pred_all.append(pred_sum / len(models))
    return row_ids_all, np.concatenate(pred_all, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    config_path = Path(args.config) if args.config else checkpoint_dir / "config.yaml"
    cfg = load_config(config_path)

    data_dir = Path(args.data_dir)
    input_dir = Path(args.input_dir) if args.input_dir else data_dir / "test_soundscapes"
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    labels = [str(c) for c in sample.columns[1:]]
    cfg["model"]["num_classes"] = len(labels)

    paths = sorted(input_dir.glob("*.ogg")) if input_dir.exists() else []
    if not paths:
        fallback = data_dir / "train_soundscapes"
        paths = sorted(fallback.glob("*.ogg"))[: int(args.dry_run_files)]
        print(f"No test soundscapes found; dry-run files={len(paths)}")
    if not paths:
        raise FileNotFoundError(f"No .ogg files found in {input_dir}")

    checkpoints = find_checkpoints(checkpoint_dir, args.folds)
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    row_ids, pred = predict_all(checkpoints, paths, cfg, device, int(args.batch_size))
    pred = np.clip(pred, 0.0, 1.0)

    out = pd.DataFrame(pred, columns=labels)
    out.insert(0, "row_id", np.array(row_ids, dtype=object))

    sample_ids = set(sample["row_id"].astype(str))
    out_ids = set(out["row_id"].astype(str))
    if sample_ids == out_ids:
        out = out.set_index("row_id").loc[sample["row_id"].astype(str), labels].reset_index()
        out.columns = sample.columns
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: rows={len(out)}, checkpoints={len(checkpoints)}")


if __name__ == "__main__":
    main()

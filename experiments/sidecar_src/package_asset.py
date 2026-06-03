import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sidecar_src.utils.config import ensure_dir, load_config


def copy_code_tree(src_root: Path, dst_root: Path) -> None:
    dst = dst_root / "sidecar_src"
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(src_root, dst, ignore=ignore)


def write_entrypoints(out_dir: Path) -> None:
    (out_dir / "infer.py").write_text(
        "from sidecar_src.inference.infer import main\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    (out_dir / "masked_rank_blend.py").write_text(
        "from sidecar_src.inference.masked_rank_blend import main\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", nargs="+", default=["0"])
    parser.add_argument("--gate-csv", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    out_dir = ensure_dir(args.out_dir)
    out_root = Path(cfg["output"]["dir"])

    shutil.copy2(cfg_path, out_dir / "config.yaml")
    labels_src = Path(cfg["data"]["labels_csv"])
    shutil.copy2(labels_src, out_dir / "labels.csv")

    for fold in args.folds:
        src = out_root / f"fold{fold}" / "best.pt"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_dir / f"fold{fold}.pt")

    oof = out_root / "oof_predictions.npz"
    if oof.exists():
        shutil.copy2(oof, out_dir / "oof_predictions.npz")

    gate_candidates = []
    if args.gate_csv:
        explicit_gate = Path(args.gate_csv)
        if not explicit_gate.exists():
            raise FileNotFoundError(explicit_gate)
        gate_candidates.append(explicit_gate)
    gate_candidates.extend(sorted(out_root.glob("*_oof_gate.csv")))
    copied_gate = None
    for gate in gate_candidates:
        if gate.exists():
            copied_gate = out_dir / gate.name
            shutil.copy2(gate, copied_gate)
            break

    src_root = Path(__file__).resolve().parent
    copy_code_tree(src_root, out_dir)
    write_entrypoints(out_dir)

    labels = pd.read_csv(labels_src)
    train_audio_enabled = bool(cfg.get("train_audio", {}).get("enabled", False))
    data_sources = ["train_soundscapes"]
    if train_audio_enabled:
        data_sources.append("train_audio")
    manifest = {
        "experiment": cfg.get("experiment", "sidecar"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_classes": int(len(labels)),
        "sample_rate": int(cfg["audio"]["sample_rate"]),
        "context_seconds": float(cfg["audio"]["context_seconds"]),
        "target_seconds": float(cfg["audio"]["target_seconds"]),
        "features": ["logmel", "pcen"],
        "model": cfg["model"],
        "folds": [int(f) for f in args.folds],
        "data_sources": data_sources,
        "train_audio_enabled": train_audio_enabled,
        "oof_gate_file": copied_gate.name if copied_gate is not None else None,
        "hidden_test_labels_used": False,
        "notes": "Sidecar SED asset. Blend into anchor with masked rank correction.",
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote asset directory: {out_dir}")


if __name__ == "__main__":
    main()

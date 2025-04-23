import sys
import logging
import shutil
from pathlib import Path

import pytest
import pandas as pd

import configure
import process

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    # Setup temporary data roots
    root = tmp_path / "test"
    data_dir = root / "data"
    audio_dir = data_dir / "train_audio" / "41970"
    soundscape_dir = data_dir / "train_soundscapes"
    audio_dir.mkdir(parents=True)
    soundscape_dir.mkdir(parents=True)

    # Copy sample .ogg files from dataset
    src_audio = Path("/data/birdclef/train_audio/41970/iNat327629.ogg")
    src_sc = Path("/data/birdclef/train_soundscapes/H02_20230420_074000.ogg")
    shutil.copy(src_audio, audio_dir / "iNat327629.ogg")
    shutil.copy(src_sc, soundscape_dir / "H02_20230420_074000.ogg")

    # Create train.csv and taxonomy.csv
    train_csv = data_dir / "train.csv"
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"primary_label": "41970", "filename": "41970/iNat327629.ogg"}
    ]).to_csv(train_csv, index=False)
    tax_csv = data_dir / "taxonomy.csv"
    pd.DataFrame([{"primary_label": "41970"}]).to_csv(tax_csv, index=False)

    # Monkeypatch CFG paths
    monkeypatch.setattr(configure.CFG, 'TRAIN_AUDIO_DIR', data_dir / 'train_audio')
    monkeypatch.setattr(configure.CFG, 'TRAIN_SOUNDSCAPE_DIR', soundscape_dir)
    monkeypatch.setattr(configure.CFG, 'TRAIN_CSV', train_csv)
    monkeypatch.setattr(configure.CFG, 'TAXONOMY_CSV', tax_csv)
    processed_dir = root / 'processed'
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed_dir)
    # No benchmark model
    monkeypatch.setattr(configure.CFG, 'BENCHMARK_MODEL', None)

    return root


def test_process_pipeline(setup_test_env, caplog):
    caplog.set_level(logging.INFO)
    # Run preprocessing
    process.main()

    proc = configure.CFG.PROCESSED_DIR

    # Train recordings outputs
    mel_train = proc / 'mels' / 'train' / '41970'
    lbl_train = proc / 'labels' / 'train' / '41970'
    assert mel_train.exists() and any(mel_train.iterdir()), f"No mel files in {mel_train}"
    assert lbl_train.exists() and any(lbl_train.iterdir()), f"No label files in {lbl_train}"

    # Soundscape outputs
    mel_sc = proc / 'mels' / 'soundscape'
    lbl_sc = proc / 'labels' / 'soundscape'
    assert mel_sc.exists() and any(mel_sc.iterdir()), f"No soundscape mels in {mel_sc}"
    assert lbl_sc.exists() and any(lbl_sc.iterdir()), f"No soundscape labels in {lbl_sc}"

    # Metadata files
    meta_train = proc / 'train_metadata.csv'
    meta_sc = proc / 'soundscape_metadata.csv'
    assert meta_train.exists(), "train_metadata.csv missing"
    assert meta_sc.exists(), "soundscape_metadata.csv missing"

    # Log check
    assert "Processing labelled recordings" in caplog.text

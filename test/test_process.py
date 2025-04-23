import sys
from pathlib import Path
# Add project root (one level up) to sys.path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import shutil

import pytest
import pandas as pd
import soundfile as sf
import logging

import configure
import process


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, caplog, monkeypatch):
    # Create test root directories under /test
    root = tmp_path / "test"
    root.mkdir()
    # Data directories
    data_dir = root / "data"
    audio_dir = data_dir / "train_audio" / "41970"
    audio_dir.mkdir(parents=True)
    soundscape_dir = data_dir / "train_soundscapes"
    soundscape_dir.mkdir(parents=True)

    # Copy real test audio and soundscape
    shutil.copy(
        Path("/data/birdclef/train_audio/41970/iNat327629.ogg"),
        audio_dir / "iNat327629.ogg"
    )
    shutil.copy(
        Path("/data/birdclef/train_soundscapes/H02_20230420_074000.ogg"),
        soundscape_dir / "H02_20230420_074000.ogg"
    )

    # Create train.csv and taxonomy.csv
    train_csv = data_dir / "train.csv"
    train_csv.parent.mkdir(parents=True)
    pd.DataFrame([
        {"primary_label": "41970", "filename": "41970/iNat327629.ogg"}
    ]).to_csv(train_csv, index=False)
    tax_csv = data_dir / "taxonomy.csv"
    pd.DataFrame([{"primary_label": "41970"}]).to_csv(tax_csv, index=False)

    # Configure CFG paths
    monkeypatch.setattr(configure.CFG, 'TRAIN_AUDIO_DIR', data_dir / 'train_audio')
    monkeypatch.setattr(configure.CFG, 'TRAIN_SOUNDSCAPE_DIR', soundscape_dir)
    monkeypatch.setattr(configure.CFG, 'TRAIN_CSV', train_csv)
    monkeypatch.setattr(configure.CFG, 'TAXONOMY_CSV', tax_csv)
    processed_dir = root / 'processed'
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed_dir)
    monkeypatch.setattr(configure.CFG, 'BENCHMARK_MODEL', None)
    processed_dir.mkdir()

    # Redirect logs to file under /test/log
    log_dir = root / 'log'
    log_dir.mkdir()
    log_file = log_dir / 'process.log'
    handler = logging.FileHandler(str(log_file))
    logging.getLogger().addHandler(handler)
    caplog.set_level('INFO')

    return {'root': root, 'processed_dir': processed_dir, 'log_file': log_file}


def test_process_run(setup_test_env, caplog):
    env = setup_test_env
    # Run preprocessing
    process.main()

    processed = env['processed_dir']
    # Check train audio outputs
    mel_train = processed / 'mels' / 'train' / '41970'
    label_train = processed / 'labels' / 'train' / '41970'
    assert mel_train.exists() and any(mel_train.iterdir()), f"No mel files in {mel_train}"
    assert label_train.exists() and any(label_train.iterdir()), f"No label files in {label_train}"

    # Check soundscape pseudo-label outputs
    mel_sound = processed / 'mels' / 'soundscape'
    label_sound = processed / 'labels' / 'soundscape'
    assert mel_sound.exists() and any(mel_sound.iterdir()), f"No soundscape mels in {mel_sound}"
    assert label_sound.exists() and any(label_sound.iterdir()), f"No soundscape labels in {label_sound}"

    # Check metadata files
    meta_train = processed / 'train_metadata.csv'
    meta_sound = processed / 'soundscape_metadata.csv'
    assert meta_train.exists(), "train_metadata.csv missing"
    assert meta_sound.exists(), "soundscape_metadata.csv missing"

    # Check log content
    log_text = env['log_file'].read_text()
    assert "Processing labelled recordings" in log_text

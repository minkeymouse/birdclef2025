import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configure
import diffwave


@pytest.fixture(autouse=True)
def setup_diffwave_env(tmp_path, monkeypatch):
    # Create test root and data directories
    root = tmp_path / 'test'
    root.mkdir()

    # Raw audio structure
    data_dir = root / 'data'
    audio_dir = data_dir / 'train_audio' / '41970'
    audio_dir.mkdir(parents=True)
    # Create placeholder audio file
    (audio_dir / 'iNat327629.ogg').write_bytes(b'')

    # Create train.csv for plan
    train_csv = data_dir / 'train.csv'
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {'primary_label': '41970', 'filename': 'iNat327629.ogg'}
    ]).to_csv(train_csv, index=False)

    # Monkey-patch CFG to point at our dirs
    monkeypatch.setattr(configure.CFG, 'TRAIN_AUDIO_DIR', data_dir / 'train_audio')
    monkeypatch.setattr(configure.CFG, 'TRAIN_CSV', train_csv)
    processed_dir = root / 'processed'
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed_dir)
    processed_dir.mkdir()

    # Create dummy mel-spectrograms under processed_dir/mels/train/41970/*
    mel_dir = processed_dir / 'mels' / 'train' / '41970'
    mel_dir.mkdir(parents=True, exist_ok=True)
    dummy_mel = np.random.rand(configure.CFG.N_MELS, 10).astype(np.float32)
    np.save(mel_dir / 'chunk0.npy', dummy_mel)

    # Stub out the actual DiffWaveVocoder so we don't need real model
    class DummyVocoder:
        @classmethod
        def from_hparams(cls, *args, **kwargs):
            return cls()
        def decode_spectrogram(self, mel_tensor, hop_length, fast_sampling, fast_sampling_noise_schedule):
            # Return a silent waveform whose length matches frames*hop_length
            frames = mel_tensor.shape[-1]
            length = frames * hop_length
            return torch.zeros((1, length), dtype=torch.float32)

    monkeypatch.setattr(diffwave, 'DiffWaveVocoder', DummyVocoder)

    return {
        'audio_dir': audio_dir,
        'train_csv': train_csv
    }


def test_diffwave_generate_and_patch(setup_diffwave_env, monkeypatch):
    ctx = setup_diffwave_env

    # Simulate CLI invocation
    monkeypatch.setattr(sys, 'argv', ['diffwave.py', 'generate'])
    diffwave.main()

    # Expect exactly one synthetic file in the species directory
    out_file = ctx['audio_dir'] / 'synthetic_000.ogg'
    assert out_file.exists(), f"Expected synthetic file at {out_file}"

    # Check that train.csv has been patched with the new entry
    df = pd.read_csv(ctx['train_csv'])
    entries = set(zip(df['primary_label'], df['filename']))
    assert ('41970', 'synthetic_000.ogg') in entries, \
        "New synthetic entry not found in train.csv"

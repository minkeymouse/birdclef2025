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
    data_dir = root / 'data'
    audio_dir = data_dir / 'train_audio' / '41970'
    audio_dir.mkdir(parents=True)
    # Create a placeholder existing audio entry for plan calculation
    (audio_dir / 'iNat327629.ogg').write_bytes(b'')

    # Create train.csv for plan
    train_csv = data_dir / 'train.csv'
    train_csv.parent.mkdir(parents=True)
    pd.DataFrame([
        {'primary_label': '41970', 'filename': 'iNat327629.ogg'}
    ]).to_csv(train_csv, index=False)

    # Set configuration paths
    monkeypatch.setattr(configure.CFG, 'TRAIN_AUDIO_DIR', data_dir / 'train_audio')
    monkeypatch.setattr(configure.CFG, 'TRAIN_CSV', train_csv)
    processed_dir = root / 'processed'
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed_dir)
    processed_dir.mkdir()

    # Create dummy mel spectrograms
    mel_dir = processed_dir / 'mels' / 'train' / '41970'
    mel_dir.mkdir(parents=True)
    # 80 mel bins, 625 frames for 5s @32kHz hop=256
    dummy_mel = np.random.rand(80, 625).astype(np.float32)
    np.save(mel_dir / 'chunk0.npy', dummy_mel)

    # Monkeypatch DiffWaveVocoder to stub decode_spectrogram
    class DummyVocoder:
        @classmethod
        def from_hparams(cls, *args, **kwargs):
            return cls()
        def decode_spectrogram(self, mel_tensor, hop_length, fast_sampling, fast_sampling_noise_schedule):
            # Return a waveform of shape [1, time]
            frames = mel_tensor.shape[-1]
            # time = frames * hop_length
            time = frames * hop_length
            return torch.zeros((1, time), dtype=torch.float32)
    monkeypatch.setattr(diffwave, 'DiffWaveVocoder', DummyVocoder)

    # Return context
    return {'root': root, 'audio_dir': audio_dir, 'processed_dir': processed_dir, 'train_csv': train_csv}


def test_diffwave_generate(setup_diffwave_env, monkeypatch):
    ctx = setup_diffwave_env
    # Simulate CLI args
    monkeypatch.setattr(sys, 'argv', ['diffwave.py', 'generate'])
    # Run diffwave generation
    diffwave.main()

    # Check that synthetic file was created
    out_file = ctx['audio_dir'] / 'synthetic_000.ogg'
    assert out_file.exists(), f"Expected synthetic file at {out_file}"

    # Check train.csv was patched
    df = pd.read_csv(ctx['train_csv'])
    rows = set(zip(df['primary_label'], df['filename']))
    assert ('41970', 'synthetic_000.ogg') in rows, "synthetic entry not found in train.csv"

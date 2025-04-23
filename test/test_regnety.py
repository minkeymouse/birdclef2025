import sys
from pathlib import Path
import json
import pytest
import torch
import numpy as np
import pandas as pd
import logging

# Insert project root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configure
import regnety


def pytest_configure():
    # ensure reproducibility
    torch.manual_seed(0)

@pytest.fixture(autouse=True)
def setup_regnety_env(tmp_path, monkeypatch, caplog):
    # Create test root
    root = tmp_path / 'test'
    root.mkdir()
    # Create processed metadata directory and sample mel-file
    processed = root / 'processed'
    processed.mkdir()
    # train_metadata.csv
    train_meta = pd.DataFrame([
        {
            'mel_path': 'mels/train/sp1/chunk0.npy',
            'label_json': json.dumps({'sp1': 1.0})
        }
    ])
    train_meta.to_csv(processed / 'train_metadata.csv', index=False)
    # soundscape_metadata.csv
    sound_meta = pd.DataFrame([
        {
            'mel_path': 'mels/soundscape/chunk0.npy',
            'label_json': json.dumps({'sp1': 1.0})
        }
    ])
    sound_meta.to_csv(processed / 'soundscape_metadata.csv', index=False)

    # Monkeypatch configuration paths
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed)
    model_dir = root / 'models' / 'regnety'
    monkeypatch.setattr(configure.CFG, 'REG_MODEL_DIR', model_dir)
    model_dir.mkdir(parents=True)

    # Stub MelDataset to return one sample
    class DummyMelDataset(torch.utils.data.Dataset):
        def __init__(self, df, s2i, augment): pass
        def __len__(self): return 1
        def __getitem__(self, idx):
            frames = regnety.SEG_SECONDS * regnety.SAMPLE_RATE // regnety.HOP_LENGTH
            x = torch.zeros((1, regnety.N_MELS, frames), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            w = torch.tensor(1.0, dtype=torch.float32)
            return x, y, w
    monkeypatch.setattr(regnety, 'MelDataset', DummyMelDataset)

    # Stub FileWiseSampler
    monkeypatch.setattr(regnety, 'FileWiseSampler', lambda df, src: list(range(len(df))))

    # Stub timm.create_model to lightweight model
    class DummyRegNet(torch.nn.Module):
        def __init__(self, arch, pretrained, in_chans, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(1, num_classes)
        def forward(self, x):
            batch = x.shape[0]
            return torch.zeros((batch, num_classes), dtype=torch.float32)
    # find model name stub
    monkeypatch.setattr(regnety.timm, 'list_models', lambda: ['regnety_008', 'regnety_008gf'])
    monkeypatch.setattr(regnety.timm, 'create_model', lambda arch, pretrained, in_chans, num_classes: DummyRegNet(arch, pretrained, in_chans, num_classes))

    # Override hyperparams for quick test
    monkeypatch.setattr(configure.CFG, 'REG_BATCH_SIZE', 1)
    monkeypatch.setattr(configure.CFG, 'REG_NUM_WORKERS', 0)
    monkeypatch.setattr(configure.CFG, 'REG_NUM_MODELS', 1)
    monkeypatch.setattr(configure.CFG, 'REG_EPOCHS', 1)
    monkeypatch.setattr(configure.CFG, 'REG_LR', 1e-3)
    monkeypatch.setattr(configure.CFG, 'REG_WEIGHT_DECAY', 0)

    # Capture logs
    caplog.set_level(logging.INFO)
    return {'processed': processed, 'model_dir': model_dir}


def test_regnety_training(setup_regnety_env, caplog, monkeypatch):
    ctx = setup_regnety_env
    # Simulate CLI args
    monkeypatch.setattr(sys, 'argv', ['regnety.py', '--device', 'cpu'])
    # Run training main
    regnety.main()

    # Check checkpoint file exists
    files = list(ctx['model_dir'].glob('regnety_008_run1.pth'))
    assert files, f"Checkpoint not found in {ctx['model_dir']}"

    # Load and inspect checkpoint
    ckpt = torch.load(files[0], map_location='cpu')
    assert 'arch' in ckpt and 'model' in ckpt and 'species2idx' in ckpt

    # species2idx mapping
    assert ckpt['species2idx'] == {'sp1': 0}

    # Confirm finish log
    assert 'Finished' in caplog.text

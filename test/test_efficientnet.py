import sys
from pathlib import Path
import json

import pytest
import numpy as np
import torch
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configure
import efficientnet


def pytest_configure():
    # Ensure deterministic behavior
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def setup_efficientnet_env(tmp_path, monkeypatch):
    # Create test root
    root = tmp_path / 'test'
    root.mkdir()
    # Processed metadata directory
    processed = root / 'processed'
    processed.mkdir()

    # Create minimal train_metadata.csv
    train_meta = pd.DataFrame([
        {
            'mel_path': 'mels/train/sp1/chunk0.npy',
            'label_json': json.dumps({'sp1': 1.0})
        }
    ])
    train_meta.to_csv(processed / 'train_metadata.csv', index=False)
    # No soundscape for this test

    # Monkeypatch CFG paths
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed)
    monkeypatch.setattr(configure.CFG, 'EFF_MODEL_DIR', root / 'models' / 'efficientnet')
    (configure.CFG.EFF_MODEL_DIR).mkdir(parents=True)

    # Stub MelDataset to yield one sample
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, df, s2i, augment): pass
        def __len__(self): return 1
        def __getitem__(self, idx):
            # x: [1,80,frames], y: long idx, w: float weight
            frames = efficientnet.SEG_SECONDS * efficientnet.SAMPLE_RATE // efficientnet.HOP_LENGTH
            x = torch.zeros((1, efficientnet.N_MELS, frames), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            w = torch.tensor(1.0, dtype=torch.float32)
            return x, y, w
    monkeypatch.setattr(efficientnet, 'MelDataset', DummyDataset)

    # Stub FileWiseSampler to simple sequential sampler
    monkeypatch.setattr(efficientnet, 'FileWiseSampler', lambda df, src: list(range(len(df))))

    # Stub timm.create_model to lightweight model
    class DummyModel(torch.nn.Module):
        def __init__(self, in_chans, num_classes):
            super().__init__()
            self.linear = torch.nn.Linear(1, num_classes)
        def forward(self, x):
            batch = x.shape[0]
            return torch.zeros((batch, len({'sp1'})), dtype=torch.float32)
    monkeypatch.setattr(efficientnet.timm, 'create_model',
                        lambda name, pretrained, in_chans, num_classes:
                        DummyModel(in_chans, num_classes))

    # Override training hyperparams for speed
    monkeypatch.setattr(configure.CFG, 'EFF_BATCH_SIZE', 1)
    monkeypatch.setattr(configure.CFG, 'EFF_NUM_WORKERS', 0)
    monkeypatch.setattr(configure.CFG, 'EFF_NUM_MODELS', 1)
    monkeypatch.setattr(configure.CFG, 'EFF_EPOCHS', 1)
    monkeypatch.setattr(configure.CFG, 'EFF_LR', 1e-3)
    monkeypatch.setattr(configure.CFG, 'EFF_WEIGHT_DECAY', 0)

    return {'root': root}


def test_efficientnet_training(setup_efficientnet_env, caplog, monkeypatch):
    caplog.set_level('INFO')
    # Simulate CLI invocation for CPU and 1 epoch
    monkeypatch.setattr(sys, 'argv', ['efficientnet.py', '--device', 'cpu', '--epochs', '1'])

    # Run main training function
    efficientnet.main()

    # Check that checkpoint file exists
    model_dir = configure.CFG.EFF_MODEL_DIR
    ckpt_files = list(model_dir.glob('efficientnet_b0_run1.pth'))
    assert ckpt_files, f"No checkpoint found in {model_dir}"

    # Load checkpoint to verify structure
    ckpt = torch.load(ckpt_files[0], map_location='cpu')
    assert 'model' in ckpt and 'species2idx' in ckpt, "Checkpoint missing keys"

    # Verify that species mapping contains our class
    assert ckpt['species2idx'] == {'sp1': 0}

    # Check log output contains finish message
    assert 'Finished' in caplog.text

import sys
from pathlib import Path
import json
import random
import numpy as np
import torch
import pandas as pd
import pytest

# ensure project root import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configure
import data_utils as du

@pytest.fixture(autouse=True)
def deterministic_seed():
    # Ensure reproducible randomness
    du.seed_everything(123)
    return None

def test_seed_everything():
    # Test that random, numpy, and torch seeds are consistent
    du.seed_everything(42)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(1).item()

    du.seed_everything(42)
    assert random.random() == pytest.approx(a)
    assert np.random.rand() == pytest.approx(b)
    assert torch.rand(1).item() == pytest.approx(c)

def test_compute_noise_metric():
    # Create a known array
    y = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    # compute std, var, rms, sum
    expected = float(y.std() + y.var() + np.sqrt((y**2).mean()) + (y**2).sum())
    assert du.compute_noise_metric(y) == pytest.approx(expected)

def test_spec_augment_and_cutmix():
    # Create dummy mel array
    mel = np.ones((10, 20), dtype=np.float32)
    # Spec augment should maintain shape
    mel_aug = du.spec_augment(mel, freq_mask_param=2, time_mask_param=3, num_masks=1)
    assert mel_aug.shape == mel.shape
    # At least some zeros appear in augmented mel
    assert np.any(mel_aug == 0.0)

    # Test cutmix merges two arrays
    m1 = np.ones((5, 10), dtype=np.float32)
    l1 = torch.tensor([1.0, 0.0])  # dummy labels length mismatch but we care dimensions
    m2 = np.zeros((5, 10), dtype=np.float32)
    l2 = torch.tensor([0.5, 0.5])
    mixed, label = du.cutmix(m1, l1, m2, l2)
    # mixed shape same
    assert mixed.shape == m1.shape
    # label shape matches first target shape
    assert isinstance(label, torch.Tensor)

def test_segment_audio():
    # Create sine wave of length 2 seconds at 1 Hz sample
    sr = 10
    y = np.arange(sr * 2).astype(np.float32)
    # monkeypatch CFG for chunk/hop secs
    monkey_cfg = configure.CFG
    monkey_cfg.TRAIN_CHUNK_SEC = 1
    monkey_cfg.TRAIN_CHUNK_HOP_SEC = 1
    monkey_cfg.SAMPLE_RATE = sr
    segments = list(du.segment_audio(y, sr=sr))
    # Expect 2 segments (start 0,1)
    assert len(segments) == 2
    for start_sec, chunk in segments:
        assert len(chunk) == sr * 1

def test_load_and_trim_audio(monkeypatch):
    # Stub librosa.load and trim functions
    called = {}
    def fake_load(fp, sr, mono=True):
        called['load'] = (fp, sr, mono)
        return np.array([0.1, -0.1], dtype=np.float32), sr
    monkeypatch.setattr(du.librosa, 'load', fake_load)
    def fake_trim(y, top_db):
        called['trim'] = (y.tolist(), top_db)
        return y[:1], (0,)
    monkeypatch.setattr(du.librosa.effects, 'trim', fake_trim)

    y = du.load_audio('dummy.wav', sample_rate=16000)
    assert isinstance(y, np.ndarray)
    y_trim = du.trim_silence(y)
    assert isinstance(y_trim, np.ndarray)
    assert 'load' in called and 'trim' in called

def test_compute_mel_cpu(monkeypatch):
    # Force CPU path
    monkeypatch.setattr(du, '_HAS_SB', False)
    # Stub librosa.feature and power_to_db
    mel_out = np.ones((4,5), dtype=np.float32)
    monkeypatch.setattr(du.librosa.feature, 'melspectrogram', lambda **kw: mel_out)
    monkeypatch.setattr(du.librosa, 'power_to_db', lambda m, ref: m)
    # Override CFG params
    monkey_cfg = configure.CFG
    monkey_cfg.SAMPLE_RATE = 16000
    monkey_cfg.HOP_LENGTH = 128
    monkey_cfg.N_FFT = 512
    monkey_cfg.N_MELS = 4
    monkey_cfg.FMIN = 0
    monkey_cfg.FMAX = 8000
    monkey_cfg.POWER = 2.0
    # Dummy audio
    y = np.random.randn(100).astype(np.float32)
    mel = du.compute_mel(y, to_db=True)
    assert np.allclose(mel, mel_out)

def test_FileWiseSampler():
    # Create DataFrame with two file groups
    df = pd.DataFrame({'filepath': ['a.wav','a.wav','b.wav']})
    sampler = du.FileWiseSampler(df, 'filepath')
    indices = list(iter(sampler))
    # Expect two indices, one for each file
    assert len(indices) == 2
    # Indices should be within valid range
    assert set(indices).issubset({0,1,2})

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self): self.df = pd.DataFrame({'mel_path':['p1.npy'], 'label_json':[json.dumps({'sp':1.0})]})
    def __len__(self): return 1
    def __getitem__(self, idx): return None

def test_MelDataset(monkeypatch, tmp_path):
    # Prepare fake data
    processed = tmp_path / 'processed'
    processed.mkdir()
    # Save dummy mel and label .npy
    mel_dir = processed / 'mels' / 'train' / 'sp'
    label_dir = processed / 'labels' / 'train' / 'sp'
    mel_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    mel_arr = np.ones((2,2),dtype=np.float32)
    np.save(mel_dir/'p1.npy', mel_arr)
    label_vec = np.array([1.0],dtype=np.float32)
    np.save(label_dir/'p1.label.npy', label_vec)
    # Monkeypatch CFG and dependencies
    monkeypatch.setattr(configure.CFG, 'PROCESSED_DIR', processed)
    monkeypatch.setattr(configure.CFG, 'TARGET_SHAPE', (2,2))
    monkey_cfg = configure.CFG
    monkey_cfg.USE_SOFT_LABELS = True
    # Dummy cv2.resize to identity
    monkeypatch.setattr(du.cv2, 'resize', lambda img, shape, interpolation: img)
    df = pd.DataFrame({
        'mel_path': ['mels/train/sp/p1.npy'],
        'label_path': ['labels/train/sp/p1.label.npy'],
    })
    ds = du.MelDataset(df, {'sp':0}, augment=False)
    mel_tensor, label, weight = ds[0]
    # Check shapes/types
    assert isinstance(mel_tensor, torch.Tensor)
    assert mel_tensor.shape[1:] == mel_arr.shape
    assert isinstance(label, torch.Tensor)
    assert weight == 1.0

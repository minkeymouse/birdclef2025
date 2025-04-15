#!/usr/bin/env python
"""
BirdCLEF 2025 Training Pipeline

This script implements the training pipeline for BirdCLEF 2025.
It includes data cleaning (with voice removal), data augmentation,
mel spectrogram extraction, and training of a CNN model (an EfficientNet-B0 sample).

Usage:
  python train_birdclef2025.py train   # to train the model

IMPORTANT:
To ensure training continues after disconnecting from the server, run this script 
within a persistent session such as tmux, screen, or using nohup. For example:

    tmux new -s birdclef_training
    python train_birdclef2025.py train

or

    nohup python train_birdclef2025.py train &
"""

import os
import sys
import glob
import random
import time
import numpy as np
import pandas as pd
import librosa
import cv2
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore")

# Set up logging to both console and a log file.
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler("training_log.txt", mode='a')])
logger = logging.getLogger()

# -------------------------------
# CONFIGURATION
# -------------------------------
class CFG:
    DATA_DIR = '/data/birdclef'
    TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, 'train_audio')
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TAXONOMY_CSV = os.path.join(DATA_DIR, 'taxonomy.csv')
    # Although inference paths exist in the full pipeline, they are not used here.

    SAMPLE_RATE = 32000
    WINDOW_SIZE_SEC = 5           # For inference windows; training uses 10 sec.
    TRAIN_WINDOW_SEC = 10         # Use 10-second chunks for training.
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_IMG_SIZE = (256, 256)
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 206  # This will be updated based on the taxonomy.
    
    SEED = 42
    DEBUG = False
    
    # Inference parameters are not used in training.
    use_tta = False  
    tta_count = 3
    threshold = 0.7
    
    use_latent = True  # Latent predictions are typically used only during inference.

CFG = CFG()

def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

# -------------------------------
# DATA LOADING & PREPROCESSING FUNCTIONS
# -------------------------------
def load_audio(path, sr=CFG.SAMPLE_RATE):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def compute_mel_spectrogram(audio, sr=CFG.SAMPLE_RATE):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS,
        fmin=CFG.FMIN,
        fmax=CFG.FMAX,
        power=2.0)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_norm

def resize_spectrogram(melspec, target_size=CFG.TARGET_IMG_SIZE):
    melspec_img = cv2.resize(melspec, target_size, interpolation=cv2.INTER_LINEAR)
    return melspec_img.astype(np.float32)

# -------------------------------
# Voice Activity Detection (VAD)
# -------------------------------
def apply_vad(audio, sr=CFG.SAMPLE_RATE):
    try:
        vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        get_speech_timestamps = utils[0]
    except Exception as e:
        logger.error(f"Error loading Silero VAD: {e}")
        return audio
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    # Pass vad_model as required by the API
    speech_ts = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sr)
    mask = np.ones(len(audio))
    for seg in speech_ts:
        mask[seg['start']:seg['end']] = 0
    cleaned = audio * mask
    return cleaned

# -------------------------------
# Augmentation Functions
# -------------------------------
def augment_noise_injection(audio, sr=CFG.SAMPLE_RATE, noise_level=(0, 0.5)):
    noise = np.random.randn(len(audio))
    level = np.random.uniform(*noise_level)
    return audio + level * noise

def augment_time_shift(audio, sr=CFG.SAMPLE_RATE, shift_max_sec=2):
    shift = np.random.randint(sr * shift_max_sec)
    direction = np.random.choice([1, -1])
    shifted = np.roll(audio, direction * shift)
    if direction == 1:
        shifted[:shift] = 0
    else:
        shifted[-shift:] = 0
    return shifted

def augment_pitch_shift(audio, sr=CFG.SAMPLE_RATE, n_steps_range=(-2, 2)):
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def augment_time_stretch(audio, sr=CFG.SAMPLE_RATE, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(y=audio, rate=rate)

def random_augmentation(audio, sr=CFG.SAMPLE_RATE):
    funcs = [augment_noise_injection, augment_time_shift, augment_pitch_shift, augment_time_stretch]
    func = random.choice(funcs)
    return func(audio, sr)

# -------------------------------
# Dataset Class
# -------------------------------
class BirdClefDataset(Dataset):
    def __init__(self, df, audio_dir, species2idx, mode='train', augment=False, use_vad=True, segment_sec=CFG.TRAIN_WINDOW_SEC):
        self.df = df.copy().reset_index(drop=True)
        self.audio_dir = audio_dir
        self.species2idx = species2idx  # mapping from species id (string) to index
        self.mode = mode
        self.augment = augment
        self.use_vad = use_vad
        self.segment_sec = segment_sec
        self.sr = CFG.SAMPLE_RATE
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        try:
            audio = load_audio(file_path, sr=self.sr)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            audio = np.zeros(int(self.sr * self.segment_sec))
        if self.use_vad and 'Fabio A. Sarria-S' in str(row.get('author', '')):
            audio = apply_vad(audio, sr=self.sr)
        if self.mode == 'train' and self.augment:
            audio = random_augmentation(audio, sr=self.sr)
        required_length = int(self.sr * self.segment_sec)
        if len(audio) < required_length:
            audio = np.pad(audio, (0, required_length - len(audio)), mode='wrap')
        else:
            audio = audio[:required_length]
        melspec = compute_mel_spectrogram(audio, sr=self.sr)
        melspec = resize_spectrogram(melspec, target_size=CFG.TARGET_IMG_SIZE)
        melspec = np.expand_dims(melspec, axis=0)
        
        label = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        if 'label' in row and pd.notnull(row['label']):
            sp = str(row['label']).strip()
            if sp in self.species2idx:
                label[self.species2idx[sp]] = 1.0
            else:
                logger.error(f"Label {sp} not found in mapping!")
        elif 'secondary_labels' in row and pd.notnull(row['secondary_labels']):
            parts = str(row['secondary_labels']).strip("[]").replace("'", "").split(',')
            for part in parts:
                sp = part.strip()
                if not sp:
                    continue
                if sp in self.species2idx:
                    label[self.species2idx[sp]] = 1.0
                else:
                    logger.error(f"Secondary label {sp} not found in mapping!")
        return torch.tensor(melspec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -------------------------------
# Model Definition
# -------------------------------
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes=CFG.NUM_CLASSES, backbone_name='efficientnet_b0', in_channels=1):
        super(BirdCLEFModel, self).__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=in_channels,
                                          drop_rate=0.0, drop_path_rate=0.0)
        if hasattr(self.backbone, 'classifier'):
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        if len(feat.shape) == 4:
            feat = self.pooling(feat)
            feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits

# -------------------------------
# Training & Validation Functions
# -------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device=CFG.DEVICE):
    model.train()
    losses = []
    for inputs, targets in tqdm(loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate_epoch(model, loader, criterion, device=CFG.DEVICE):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
    return np.mean(losses)

# -------------------------------
# MAIN: Training Entry Point
# -------------------------------
def main():
    # We only focus on training in this script.
    logger.info("Loading training CSV...")
    df_train = pd.read_csv(CFG.TRAIN_CSV)
    df_train = df_train.drop_duplicates(subset='filename').reset_index(drop=True)
    
    # Build species mapping using taxonomy CSV (fallback to submission header if taxonomy not available).
    try:
        taxonomy_df = pd.read_csv(CFG.TAXONOMY_CSV)
        species_ids = taxonomy_df['primary_label'].astype(str).tolist()
    except Exception as e:
        logger.error("Error reading taxonomy CSV; falling back to sample submission header.")
        sample_sub = pd.read_csv(CFG.SUBMISSION_CSV)
        species_ids = list(sample_sub.columns)[1:]
    
    species2idx = {sp: idx for idx, sp in enumerate(species_ids)}
    CFG.NUM_CLASSES = len(species_ids)
    logger.info(f"Number of classes (after mapping): {CFG.NUM_CLASSES}")
    
    dataset = BirdClefDataset(df_train, CFG.TRAIN_AUDIO_DIR, species2idx, mode='train', augment=True, use_vad=True)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
    
    device = CFG.DEVICE
    model = BirdCLEFModel(CFG, backbone_name='efficientnet_b0').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = np.inf
    
    total_start = time.time()
    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{CFG.NUM_EPOCHS}")
        epoch_start = time.time()
        
        train_loss = train_one_epoch(model, loader, optimizer, criterion, device=device)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        val_loss = validate_epoch(model, loader, criterion, device=device)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        epoch_duration = time.time() - epoch_start
        remaining_time = (CFG.NUM_EPOCHS - epoch) * epoch_duration
        logger.info(f"Epoch duration: {epoch_duration:.2f} sec, Estimated remaining time: {remaining_time/60:.2f} min")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, 'best_model.pth')
            logger.info("Saved Best Model.")
            
    total_duration = time.time() - total_start
    logger.info(f"\nTraining completed in {total_duration/60:.2f} minutes.")

if __name__ == '__main__':
    main()

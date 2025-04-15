#!/usr/bin/env python
"""
BirdCLEF 2025 Winning Pipeline

This script implements an end-to-end solution for BirdCLEF 2025,
including data cleaning (with voice removal), augmentation,
mel spectrogram extraction, training of a CNN model (EfficientNet-B0 sample),
a latent space decomposition module, and sliding-window ensemble inference.

Usage:
  python birdclef_2025_solution.py train   # to train a model
  python birdclef_2025_solution.py infer   # to run inference and generate submission.csv

IMPORTANT:
To ensure training continues after disconnecting from the server,
run this script within a persistent session such as tmux/screen or via nohup:
  tmux new -s birdclef_training
  python birdclef_2025_solution.py train
or
  nohup python birdclef_2025_solution.py train &
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
from scipy.optimize import nnls
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# CONFIGURATION
# -------------------------------
class CFG:
    # Root directory for BirdCLEF 2025 dataset
    DATA_DIR = '/data/birdclef'
    TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, 'train_audio')
    TRAIN_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, 'train_soundscapes')  # unlabeled, if used for pseudo-labeling
    TEST_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, 'test_soundscapes')
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TAXONOMY_CSV = os.path.join(DATA_DIR, 'taxonomy.csv')  # if available
    SUBMISSION_CSV = os.path.join(DATA_DIR, 'sample_submission.csv')
    
    # Audio parameters
    SAMPLE_RATE = 32000
    WINDOW_SIZE_SEC = 5            # For inference: each prediction is on a 5-sec window.
    TRAIN_WINDOW_SEC = 10          # For training, we use a 10-sec chunk.
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_IMG_SIZE = (256, 256)   # For CNN input.
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Number of species classes – if taxonomy.csv is available, we can derive it.
    NUM_CLASSES = 206
    
    SEED = 42
    DEBUG = False  # set to True to use only a few files during inference
    
    # Inference hyperparameters (for TTA etc.)
    use_tta = False  
    tta_count = 3
    threshold = 0.7
    
    # Option to integrate latent-space predictions
    use_latent = True

CFG = CFG()

# -------------------------------
# SEED SETUP
# -------------------------------
def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

# -------------------------------
# DATA LOADING & PREPROCESSING
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
    """
    Use Silero VAD to remove human speech.
    For recordings by known authors (e.g. "Fabio A. Sarria-S"), we mute segments with detected speech.
    """
    try:
        vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        get_speech_timestamps = utils[0]
    except Exception as e:
        print("Error loading Silero VAD:", e)
        return audio
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    speech_ts = get_speech_timestamps(audio_tensor, sampling_rate=sr)
    mask = np.ones(len(audio))
    for seg in speech_ts:
        mask[seg['start']:seg['end']] = 0
    cleaned = audio * mask
    return cleaned

# -------------------------------
# Augmentation Functions
# -------------------------------
def augment_noise_injection(audio, noise_level=(0, 0.5)):
    noise = np.random.randn(len(audio))
    level = np.random.uniform(*noise_level)
    return audio + level * noise

def augment_time_shift(audio, shift_max_sec=2, sr=CFG.SAMPLE_RATE):
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
    return librosa.effects.pitch_shift(audio, sr, n_steps)

def augment_time_stretch(audio, sr=CFG.SAMPLE_RATE, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate)


def random_augmentation(audio, sr=CFG.SAMPLE_RATE):
    funcs = [augment_noise_injection, augment_time_shift, augment_pitch_shift, augment_time_stretch]
    func = random.choice(funcs)
    return func(audio, sr)

# -------------------------------
# Dataset Class
# -------------------------------
class BirdClefDataset(Dataset):
    def __init__(self, df, audio_dir, mode='train', augment=False, use_vad=True, segment_sec=CFG.TRAIN_WINDOW_SEC):
        self.df = df.copy().reset_index(drop=True)
        self.audio_dir = audio_dir
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
            print(f"Error loading {file_path}: {e}")
            audio = np.zeros(int(self.sr * self.segment_sec))
        if self.use_vad and 'Fabio A. Sarria-S' in str(row.get('author', '')):
            audio = apply_vad(audio, sr=self.sr)
        if self.mode == 'train' and self.augment:
            audio = random_augmentation(audio, sr=self.sr)
        required_length = int(self.sr * self.segment_sec)
        if len(audio) < required_length:
            pad_length = required_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='wrap')
        else:
            audio = audio[:required_length]
        melspec = compute_mel_spectrogram(audio, sr=self.sr)
        melspec = resize_spectrogram(melspec, target_size=CFG.TARGET_IMG_SIZE)
        melspec = np.expand_dims(melspec, axis=0)
        
        label = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        if 'label' in row and pd.notnull(row['label']):
            label[int(row['label'])] = 1.0
        elif 'secondary_labels' in row and pd.notnull(row['secondary_labels']):
            parts = str(row['secondary_labels']).strip("[]").replace("'", "").split(',')
            for part in parts:
                part = part.strip()
                if part.isdigit():
                    label[int(part)] = 1.0
        return torch.tensor(melspec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -------------------------------
# Model Definitions
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
# Latent Space Modeling Component
# -------------------------------
def build_species_dictionary(df, audio_dir, num_samples=5):
    species_dict = {}
    for species in range(CFG.NUM_CLASSES):
        specs = []
        subset = df[df['label'] == species]
        for _, row in subset.head(num_samples).iterrows():
            file_path = os.path.join(audio_dir, row['filename'])
            try:
                audio = load_audio(file_path, sr=CFG.SAMPLE_RATE)
                req_len = int(CFG.SAMPLE_RATE * CFG.WINDOW_SIZE_SEC)
                if len(audio) < req_len:
                    audio = np.pad(audio, (0, req_len - len(audio)), mode='wrap')
                else:
                    audio = audio[:req_len]
                mel = compute_mel_spectrogram(audio, sr=CFG.SAMPLE_RATE)
                mel = resize_spectrogram(mel, CFG.TARGET_IMG_SIZE)
                specs.append(mel.flatten())
            except Exception as e:
                continue
        if specs:
            species_dict[species] = np.mean(specs, axis=0)
        else:
            species_dict[species] = np.zeros(CFG.TARGET_IMG_SIZE[0] * CFG.TARGET_IMG_SIZE[1])
    species_dict['noise'] = np.zeros(CFG.TARGET_IMG_SIZE[0] * CFG.TARGET_IMG_SIZE[1])
    return species_dict

def latent_prediction(melspec, species_dict):
    D_list = []
    species_keys = []
    for key in species_dict:
        if key == 'noise':
            continue
        D_list.append(species_dict[key])
        species_keys.append(key)
    D = np.vstack(D_list).T
    x = melspec.flatten()
    weights, _ = nnls(D, x)
    if weights.sum() > 0:
        probs = weights / weights.sum()
    else:
        probs = np.zeros_like(weights)
    pred = np.zeros(CFG.NUM_CLASSES)
    for i, sp in enumerate(species_keys):
        pred[int(sp)] = probs[i]
    return pred

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
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    return np.mean(losses), torch.cat(all_outputs), torch.cat(all_targets)

# -------------------------------
# Test-Time Augmentation Function (TTA)
# -------------------------------
def apply_tta(spec, tta_idx):
    if tta_idx == 0:
        return spec
    elif tta_idx == 1:
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        return np.flip(spec, axis=0)
    return spec

# -------------------------------
# Inference: Sliding-Window & Ensemble
# -------------------------------
def predict_on_audio_file(audio_path, models, species_dict=None, use_latent=True):
    try:
        audio, _ = librosa.load(audio_path, sr=CFG.SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return {}
    window_samples = int(CFG.SAMPLE_RATE * CFG.WINDOW_SIZE_SEC)
    stride = window_samples // 2
    predictions = {}
    base_id = os.path.splitext(os.path.basename(audio_path))[0]
    for start in range(0, len(audio) - window_samples + 1, stride):
        segment = audio[start:start + window_samples]
        mel = compute_mel_spectrogram(segment, sr=CFG.SAMPLE_RATE)
        mel = resize_spectrogram(mel, target_size=CFG.TARGET_IMG_SIZE)
        tta_preds = []
        for tta_idx in range(CFG.tta_count if CFG.use_tta else 1):
            augmented = apply_tta(mel, tta_idx)
            inp = np.expand_dims(np.expand_dims(augmented, axis=0), axis=0)
            inp_tensor = torch.tensor(inp, dtype=torch.float32).to(CFG.DEVICE)
            model_preds = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    out = model(inp_tensor)
                    prob = torch.sigmoid(out).cpu().numpy().squeeze()
                    model_preds.append(prob)
            tta_preds.append(np.mean(model_preds, axis=0))
        nn_pred = np.mean(tta_preds, axis=0)
        if use_latent and species_dict is not None:
            latent_pred = latent_prediction(mel, species_dict)
            final_pred = 0.8 * nn_pred + 0.2 * latent_pred
        else:
            final_pred = nn_pred
        end_time = (start + window_samples) / CFG.SAMPLE_RATE
        row_id = f"{base_id}_{end_time:.2f}"
        predictions[row_id] = final_pred
    return predictions

# -------------------------------
# MAIN: Training and Inference Entry Points
# -------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        print("Loading training CSV...")
        df_train = pd.read_csv(CFG.TRAIN_CSV)
        df_train = df_train.drop_duplicates(subset='filename').reset_index(drop=True)
        dataset = BirdClefDataset(df_train, CFG.TRAIN_AUDIO_DIR, mode='train', augment=True, use_vad=True)
        loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
        
        device = CFG.DEVICE
        model = BirdCLEFModel(CFG, backbone_name='efficientnet_b0').to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR)
        criterion = nn.BCEWithLogitsLoss()
        best_loss = np.inf
        
        total_start = time.time()
        for epoch in range(1, CFG.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{CFG.NUM_EPOCHS}")
            epoch_start = time.time()
            
            train_loss = train_one_epoch(model, loader, optimizer, criterion, device=device)
            print(f"Train Loss: {train_loss:.4f}", flush=True)
            
            val_loss, val_outputs, val_targets = validate_epoch(model, loader, criterion, device=device)
            print(f"Validation Loss: {val_loss:.4f}", flush=True)
            
            epoch_duration = time.time() - epoch_start
            remaining_time = (CFG.NUM_EPOCHS - epoch) * epoch_duration
            print(f"Epoch {epoch} duration: {epoch_duration:.2f} sec, Estimated remaining time: {remaining_time/60:.2f} min", flush=True)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, 'best_model.pth')
                print("Saved Best Model.", flush=True)
                
        total_duration = time.time() - total_start
        print(f"\nTraining completed in {total_duration/60:.2f} minutes.", flush=True)
    
    elif mode == 'infer':
        print("Starting inference...")
        sub_df = pd.read_csv(CFG.SUBMISSION_CSV)
        species_columns = list(sub_df.columns)[1:]
        CFG.NUM_CLASSES = len(species_columns)
        model_files = glob.glob(os.path.join('models', '*.pth'))
        if not model_files:
            model_files = ['best_model.pth']
        models = []
        for ckpt in model_files:
            m = BirdCLEFModel(CFG, backbone_name='efficientnet_b0').to(CFG.DEVICE)
            checkpoint = torch.load(ckpt, map_location=CFG.DEVICE)
            m.load_state_dict(checkpoint['model_state_dict'])
            m.eval()
            models.append(m)
            print(f"Loaded model: {ckpt}")
        df_train = pd.read_csv(CFG.TRAIN_CSV)
        species_dict = build_species_dictionary(df_train, CFG.TRAIN_AUDIO_DIR, num_samples=5)
        
        test_files = glob.glob(os.path.join(CFG.TEST_SOUNDSCAPES_DIR, '*.ogg'))
        if CFG.DEBUG:
            test_files = test_files[:CFG.DEBUG]
        print(f"Found {len(test_files)} test soundscape files.")
        
        submission_dict = {}
        for test_file in tqdm(test_files, desc="Predicting Test Soundscapes"):
            preds = predict_on_audio_file(test_file, models, species_dict=species_dict, use_latent=True)
            submission_dict.update(preds)
        
        rows = []
        for row_id, pred in submission_dict.items():
            row = [row_id] + list(pred)
            rows.append(row)
        submission = pd.DataFrame(rows, columns=["row_id"] + species_columns)
        submission.to_csv("submission.csv", index=False)
        print("Inference complete. 'submission.csv' generated.", flush=True)
    
    else:
        print("Invalid mode. Use 'train' or 'infer'.")

if __name__ == '__main__':
    main()

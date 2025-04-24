# train_utils.py
import os
import yaml
import math
import numpy as np
import pandas as pd
import librosa
import cv2
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

# --- Dataset definition ---
class BirdClefDataset(Dataset):
    """Dataset for BirdCLEF audio chunks with mel spectrograms and soft labels."""
    def __init__(self, metadata_df: pd.DataFrame, class_map: dict, 
                 mel_shape=(128, 640), augment=False):
        """
        metadata_df: DataFrame with columns: mel_path (or audio_path), label_json (or labels), weight, etc.
        class_map: dict mapping species code -> class index.
        mel_shape: expected shape (mel_bins, time_steps) of spectrogram (before channel expansion).
        augment: whether to apply data augmentation to mel spectrograms.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.class_map = class_map
        self.num_classes = len(class_map)
        self.mel_shape = mel_shape
        self.augment = augment

        # Pre-extract label vectors and weights for speed
        self.labels = []
        self.sample_weights = []
        for _, row in self.df.iterrows():
            # If labels are stored as JSON string (e.g., {"sp1":0.7,"sp2":0.3}), parse it
            if 'label_json' in row and isinstance(row['label_json'], str):
                label_dict = json.loads(row['label_json'])
            else:
                # Otherwise, assume there's a column 'labels' as dict or similar
                label_dict = row.get('labels', {})
            # Initialize zero vector and fill in any present species
            label_vec = np.zeros(self.num_classes, dtype=np.float32)
            for sp, val in label_dict.items():
                if sp in self.class_map:  # only use species known in class_map
                    label_vec[self.class_map[sp]] = float(val)
            self.labels.append(label_vec)
            # Sample weight (for loss/sampling) - use 'weight' column if present, else default 1.0
            w = row.get('weight', 1.0)
            self.sample_weights.append(float(w))
        self.labels = np.array(self.labels, dtype=np.float32)
        self.sample_weights = np.array(self.sample_weights, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load precomputed mel spectrogram if available, otherwise compute from audio
        row = self.df.iloc[idx]
        # Determine source path
        if 'mel_path' in row and pd.notna(row['mel_path']):
            mel_path = row['mel_path']
            mel = np.load(mel_path)  # expecting shape (mel_bins, time_steps)
        else:
            # Compute mel from audio file path
            audio_path = row['audio_path'] if 'audio_path' in row else None
            if audio_path is None or not os.path.exists(audio_path):
                raise FileNotFoundError(f"No audio or mel file for index {idx}")
            # Load audio (mono) and compute mel spectrogram
            y, sr = librosa.load(audio_path, sr=self.sr)  # self.sr could be set from config, e.g., 32000
            # Pad or truncate to desired chunk duration
            target_len = int(self.chunk_duration * sr)
            if len(y) < target_len:
                # pad with silence (wrap around if needed for cyclic padding)
                pad_len = target_len - len(y)
                y = np.concatenate([y, np.zeros(pad_len, dtype=y.dtype)])
            else:
                y = y[:target_len]
            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, 
                                                n_mels=self.mel_shape[0], fmin=self.fmin, fmax=self.fmax)
            mel = librosa.power_to_db(mel, ref=np.max)  # convert to log-mel
            # Normalize mel (if desired), e.g., 0-1 scaling or standardization (not strictly necessary if model can learn scale)
            # mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        # Ensure mel shape matches expected, resize if needed
        if mel.shape != tuple(self.mel_shape):
            # Resize spectrogram to target shape (e.g., 256x256) using interpolation
            mel = cv2.resize(mel, (self.mel_shape[1], self.mel_shape[0]))
        # Data augmentation on mel spectrogram (time/freq masking)
        if self.augment:
            mel = self._augment_mel(mel)
        # Convert to tensor and expand to 3-channel (for CNN input)
        mel_tensor = torch.from_numpy(mel).float()
        # Channel-first format for CNN (1 x H x W), then repeat to 3 x H x W to simulate RGB
        mel_tensor = mel_tensor.unsqueeze(0)  # shape: [1, H, W]
        mel_tensor = mel_tensor.repeat(3, 1, 1)  # shape: [3, H, W]
        # Prepare label and weight
        label_vec = torch.from_numpy(self.labels[idx])
        sample_w = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return mel_tensor, label_vec, sample_w

    def _augment_mel(self, mel):
        """Apply in-place random time-frequency masking to the mel spectrogram (numpy array)."""
        H, W = mel.shape
        # Random frequency mask
        num_masks = np.random.randint(1, 3)  # e.g., 1 or 2 freq masks
        for _ in range(num_masks):
            f0 = np.random.randint(0, H)
            f_len = np.random.randint(5, H // 8)  # mask up to H/8 mel bins
            mel[f0:f0+f_len, :] = mel.mean()  # fill with mean or 0
        # Random time mask
        num_time_masks = np.random.randint(1, 3)
        for _ in range(num_time_masks):
            t0 = np.random.randint(0, W)
            t_len = np.random.randint(5, W // 8)  # mask up to W/8 time frames
            mel[:, t0:t0+t_len] = mel.mean()
        # (Additional augmentation like adding background noise, random shifts, mixup could be included as needed)
        return mel

# --- Sampler and DataLoader utility ---
def create_dataloader(dataset: BirdClefDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Create DataLoader for the given dataset, using weighted sampling if sample weights are provided."""
    if hasattr(dataset, 'sample_weights') and dataset.sample_weights is not None:
        # Use WeightedRandomSampler to sample each batch according to sample_weights
        sampler = WeightedRandomSampler(weights=dataset.sample_weights, num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                          num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=0 if os.name == 'nt' else 4, pin_memory=True)

# --- Training loop with validation ---
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                config: dict, device: torch.device):
    """
    Train the model for the specified number of epochs, evaluate on validation set, 
    and save checkpoints for top-3 validation scores.
    """
    epochs = config['training']['epochs']
    lr = config['optimizer']['lr']
    weight_decay = config['optimizer'].get('weight_decay', 0.0)
    optimizer_type = config['optimizer'].get('type', 'Adam')
    scheduler_cfg = config.get('scheduler', None)
    # Initialize optimizer (AdamW by default for better regularization)
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Initialize LR scheduler if specified
    scheduler = None
    if scheduler_cfg:
        if scheduler_cfg['type'] == 'CosineAnnealingLR':
            t_max = scheduler_cfg.get('T_max', epochs)
            eta_min = scheduler_cfg.get('eta_min', 1e-6)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_cfg['type'] == 'StepLR':
            step_size = scheduler_cfg.get('step_size', 5)
            gamma = scheduler_cfg.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # (Add other schedulers as needed)
    # Loss function: binary cross-entropy with logits (for multi-label classification)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # we'll handle reduction manually for weighting
    best_scores = []  # keep track of top validation scores (for model saving)
    best_checkpoints = []  # store corresponding state dicts or file paths

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets, sample_w = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_w = sample_w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            logits = model(inputs)
            # Compute BCE loss per sample (and per class), then weight each sample's loss
            # criterion returns shape [batch, num_classes] since reduction='none'
            loss_matrix = criterion(logits, targets)
            # Average loss over classes for each sample, then apply sample weight
            # (target may be multi-label soft, but BCE is applied elementwise)
            loss_per_sample = loss_matrix.mean(dim=1)  # mean over classes
            weighted_loss = (loss_per_sample * sample_w)
            loss = weighted_loss.mean()  # average over batch
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        if scheduler:
            scheduler.step()
        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation phase at epoch end
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, _ = batch  # no sample weight needed for evaluation
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                preds = torch.sigmoid(logits)  # convert to probabilities
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        # Compute validation score (macro-AUC skipping classes with no positives)
        val_score = macro_auc_score(all_targets, all_preds)
        # (Optionally compute other metrics like macro-precision for monitoring)
        val_precision = macro_precision_score(all_targets, all_preds, threshold=0.5)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val AUC: {val_score:.4f} - Val Prec@0.5: {val_precision:.4f}")

        # Save checkpoint if among top-3
        save_path = None
        if len(best_scores) < 3 or val_score > min(best_scores):
            # Save model state dict
            os.makedirs(config['paths']['models_dir'], exist_ok=True)
            arch_name = config['current_arch']  # set by training script
            run_id = config.get('current_run', 1)
            # Determine checkpoint file name
            ckpt_name = f"{arch_name}_run{run_id}_epoch{epoch}.pth"
            save_path = os.path.join(config['paths']['models_dir'], ckpt_name)
            torch.save({'model_state_dict': model.state_dict(), 
                        'species2idx': config['class_map']}, save_path)
            # Update best scores list
            if len(best_scores) < 3:
                best_scores.append(val_score)
                best_checkpoints.append(save_path)
            else:
                # replace worst of the top-3
                min_idx = np.argmin(best_scores)
                # Remove old worst checkpoint file if exists
                old_ckpt = best_checkpoints[min_idx]
                try:
                    os.remove(old_ckpt)
                except FileNotFoundError:
                    pass
                best_scores[min_idx] = val_score
                best_checkpoints[min_idx] = save_path
            print(f"Saved checkpoint: {save_path}")
    # End of training epochs
    # Return paths of best checkpoints
    # (Sorting by score so that best_scores[0] is highest)
    sorted_indices = np.argsort(best_scores)[::-1]
    best_checkpoints = [best_checkpoints[i] for i in sorted_indices]
    return best_checkpoints

# --- Evaluation metrics in training context (using functions from metrics.py, defined later) ---
def macro_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged ROC-AUC, skipping classes with no positive true labels."""
    num_classes = y_true.shape[1]
    aucs = []
    for i in range(num_classes):
        # Only compute AUC if class has at least one positive in ground truth
        if np.sum(y_true[:, i]) == 0:
            continue
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
        except ValueError:
            # This class might have all negatives in truth or constant predictions; skip it
            continue
    if len(aucs) == 0:
        return 0.0
    return float(np.mean(aucs))

def macro_precision_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute macro-averaged precision at the given threshold."""
    num_classes = y_true.shape[1]
    precisions = []
    for i in range(num_classes):
        # Skip classes with no positive predictions and no positives in truth
        pred_positive = y_pred[:, i] >= threshold
        true_positive = y_true[:, i] == 1
        if pred_positive.sum() == 0:
            # If model predicts none for this class:
            if true_positive.sum() == 0:
                continue  # no instances of this class at all
            else:
                precisions.append(0.0)
                continue
        precision_i = np.sum(true_positive & pred_positive) / pred_positive.sum()
        precisions.append(precision_i)
    if len(precisions) == 0:
        return 0.0
    return float(np.mean(precisions))

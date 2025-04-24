# infer_utils.py
import numpy as np
import torch
from torch import nn
import librosa

def load_model(arch_name: str, num_classes: int, checkpoint_path: str, device: torch.device):
    """Load a model architecture (EfficientNet or RegNetY) and weights from checkpoint."""
    # Create model architecture (with 1 input channel since we used mel as 1-channel expanded to 3 in data)
    if arch_name.startswith('efficientnet'):
        # EfficientNet-B0
        from torchvision import models
        model = models.efficientnet_b0(weights=None)  # no default weights, we'll load our own
        # Adjust input conv if needed (if model was trained with 3-channel input, we assume weights reflect that)
        # Our training duplicated mel to 3 channels, so model expects 3 channels.
        num_features = model.classifier[1].in_features  # EfficientNet-B0 classifier: [Dropout, Linear]
        model.classifier = nn.Linear(num_features, num_classes)
    elif arch_name.startswith('regnet'):
        from torchvision import models
        # RegNetY-800MF
        model = models.regnet_y_800mf(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")
    model = model.to(device)
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Also return the species index mapping if needed
    species_map = checkpoint.get('species2idx', None)
    return model, species_map

def predict_chunks(model: nn.Module, audio, sr, chunk_duration=5.0):
    """Split audio into consecutive chunks of length chunk_duration (seconds), and return model predictions for each chunk."""
    model.eval()
    chunk_size = int(chunk_duration * sr)
    # Pad audio to have full chunks
    pad_len = (-len(audio)) % chunk_size
    if pad_len > 0:
        audio = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)])
    num_chunks = len(audio) // chunk_size
    preds = []
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            y_chunk = audio[start:end]
            # Compute mel spectrogram for this chunk (same parameters as training)
            mel = librosa.feature.melspectrogram(y_chunk, sr=sr, n_fft=1024, hop_length=500, n_mels=128, fmin=40, fmax=15000)
            mel = librosa.power_to_db(mel, ref=np.max)
            # Resize mel to 128x256 if needed (assuming we want the same shape used in training)
            mel_resized = cv2.resize(mel, (256, 128))
            # Convert to 3-channel tensor
            mel_tensor = torch.from_numpy(mel_resized).float().unsqueeze(0)  # shape [1, 128, 256]
            mel_tensor = mel_tensor.unsqueeze(0)  # add channel dim -> [1, 1, 128, 256]
            mel_tensor = mel_tensor.repeat(1, 3, 1, 1)  # [1, 3, 128, 256]
            mel_tensor = mel_tensor.to(next(model.parameters()).device)
            # Forward pass
            logits = model(mel_tensor)
            prob = torch.sigmoid(logits).cpu().numpy()[0]  # probabilities for each class
            preds.append(prob)
    preds = np.array(preds)  # shape: (num_chunks, num_classes)
    return preds

def smooth_predictions(chunk_probs: np.ndarray, smoothing_window=5):
    """Apply temporal smoothing by averaging over a sliding window of `smoothing_window` consecutive chunks (centered)."""
    if smoothing_window < 2:
        return chunk_probs  # no smoothing
    half_win = smoothing_window // 2
    num_chunks, num_classes = chunk_probs.shape
    smoothed = np.zeros_like(chunk_probs)
    for t in range(num_chunks):
        # Determine window bounds
        start = max(0, t - half_win)
        end = min(num_chunks, t + half_win + 1)
        window_probs = chunk_probs[start:end]
        smoothed[t] = window_probs.mean(axis=0)
    return smoothed

def ensemble_predictions(preds_list: list, strategy: str = 'min_then_avg'):
    """
    Ensemble predictions from multiple models.
    `preds_list` is a list of numpy arrays (num_chunks x num_classes) from different models.
    strategy:
      - 'min_then_avg': take elementwise min across models of same architecture (if preds_list grouped by arch), then average between groups.
      - 'average': simple mean of all model predictions.
    """
    if strategy == 'average':
        # Simple average of all predictions
        combined = np.mean(np.stack(preds_list, axis=0), axis=0)
        return combined
    elif strategy == 'min_then_avg':
        # Assume preds_list contains groups of models from two architectures (e.g., first N from arch1, next M from arch2)
        # For robustness, we'll infer grouping by array shapes or lengths of list (if known how many per arch).
        # Here we'll assume first half of list is arch1, second half is arch2.
        k = len(preds_list) // 2
        arch1_preds = preds_list[:k]
        arch2_preds = preds_list[k:]
        # Elementwise min across models in each group
        arch1_min = np.minimum.reduce(arch1_preds)  # shape (num_chunks, num_classes)
        arch2_min = np.minimum.reduce(arch2_preds)
        # Now average the two architecture outputs
        combined = (arch1_min + arch2_min) / 2.0
        return combined
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")

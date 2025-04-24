# inference.py
import yaml
import os
import numpy as np
import pandas as pd
import librosa
import cv2
import torch
from src.utils.inference_utils import load_model, predict_chunks, smooth_predictions, ensemble_predictions

if __name__ == "__main__":
    # Load inference config
    with open("config/inference.yaml", 'r') as f:
        config = yaml.safe_load(f)
    audio_dir = config['paths']['test_audio_dir']
    models_dir = config['paths']['models_dir']
    output_file = config['paths']['output_file']
    taxonomy_path = config['paths'].get('taxonomy_csv', None)
    chunk_duration = config['inference']['chunk_duration']  # e.g., 10.0 (but competition evaluation is 5s, we'll handle splitting)
    chunk_hop = config['inference']['chunk_hop']           # e.g., 5.0
    smoothing_neighbors = config['inference'].get('smoothing_neighbors', 2)
    presence_threshold = config['inference'].get('presence_threshold', 0.5)
    ensemble_ckpts = config['ensemble']['checkpoints']
    ensemble_strategy = config['ensemble'].get('strategy', 'average')
    # Determine species list from taxonomy (for output columns)
    if taxonomy_path and os.path.exists(taxonomy_path):
        taxo_df = pd.read_csv(taxonomy_path)
        # assuming taxonomy_csv has a column 'primary_label' with species codes
        species_list = list(taxo_df['primary_label'])
    else:
        raise ValueError("Taxonomy file not found. Required to determine output classes.")
    num_classes = len(species_list)
    # Load all models
    models = []
    model_archs = []
    for ckpt_name in ensemble_ckpts:
        # Determine architecture name from checkpoint naming convention
        if ckpt_name.startswith("efficientnet"):
            arch = "efficientnet_b0"
        elif ckpt_name.startswith("regnety"):
            arch = "regnety_008"
        else:
            raise ValueError(f"Unknown model type in checkpoint name: {ckpt_name}")
        ckpt_path = os.path.join(models_dir, arch, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        model, species_map = load_model(arch, num_classes, ckpt_path, torch.device("cpu"))
        models.append(model)
        model_archs.append(arch)
    # Iterate over each test audio file
    results = []  # to accumulate output rows
    for filename in sorted(os.listdir(audio_dir)):
        if not filename.lower().endswith(('.wav', '.ogg', '.mp3')):
            continue
        file_path = os.path.join(audio_dir, filename)
        print(f"Processing {filename}...")
        # Load full audio (assuming reasonably short soundscapes or streaming if very long)
        y, sr = librosa.load(file_path, sr=None)
        # We will perform inference on 5-second chunks (as evaluation is per 5s)
        # If chunk_duration is 10 (like training chunks), we still need 5s resolution output.
        # So we can slide a 10s window with hop=5s to get overlapping predictions, then smooth.
        # Alternatively, use the 5s base segmentation directly with our models.
        # We'll do overlapping predictions: 10s window every 5s, then combine.
        base_hop = int(5.0 * sr)
        base_chunk = int(5.0 * sr)
        # Pad audio to multiple of 5 seconds
        pad_len = (-len(y)) % base_chunk
        if pad_len > 0:
            y = np.concatenate([y, np.zeros(pad_len, dtype=y.dtype)])
        n_segments = len(y) // base_chunk
        # Ensemble predictions: we will compute using our loaded models
        # For each model, produce predictions for each 5s segment
        all_model_preds = []
        for model, arch in zip(models, model_archs):
            # If model was trained on 10s chunks, we might do a 10s sliding window:
            # But simpler: just predict on each 5s as independent (the model might generalize to 5s though trained on 10s).
            # Alternatively, if needed, we could feed two consecutive 5s together to model (as it expects 10s).
            # However, for ensemble, using overlapping windows and smoothing achieves similar effect.
            preds = predict_chunks(model, y, sr, chunk_duration=5.0)
            all_model_preds.append(preds)
        # all_model_preds is list of arrays of shape (n_segments, num_classes)
        # Ensure all predictions align in segment indexing
        # Smooth each model's predictions over time neighbors (if configured)
        if smoothing_neighbors > 0:
            for m in range(len(all_model_preds)):
                all_model_preds[m] = smooth_predictions(all_model_preds[m], smoothing_window=(2 * smoothing_neighbors + 1))
        # Apply ensemble strategy
        combined_preds = ensemble_predictions(all_model_preds, strategy=ensemble_strategy)
        # Apply chunk-level threshold to decide presence (if needed for output as binary)
        binary_preds = (combined_preds >= presence_threshold).astype(np.int_)
        # Prepare output for each 5s interval
        file_id = os.path.splitext(filename)[0]
        for i in range(n_segments):
            row_id = f"{file_id}_{i*5}"  # assuming each segment indexed by start time in seconds
            prob_row = combined_preds[i]
            # We will output probabilities for submission (the evaluation expects probabilities for each species)
            # If binary presence is needed instead, one could output binary_preds.
            result = {"row_id": row_id}
            for j, species in enumerate(species_list):
                result[species] = prob_row[j]
            results.append(result)
    # Save results to CSV
    output_df = pd.DataFrame(results)
    # Ensure columns order: row_id first, then species columns in alphabetical (or taxonomy) order
    cols = ["row_id"] + species_list
    output_df = output_df[cols]
    output_df.to_csv(output_file, index=False)
    print(f"Saved inference results to {output_file}")

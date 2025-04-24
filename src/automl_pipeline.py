# automl_pipeline.py
import os
import yaml
import pandas as pd
import numpy as np
import subprocess
from src.utils.inference_utils import load_model, predict_chunks, smooth_predictions, ensemble_predictions
from src.utils.metrics import create_pseudo_labels

if __name__ == "__main__":
    # Load configs
    with open("config/train.yaml", 'r') as f:
        train_config = yaml.safe_load(f)
    with open("config/inference.yaml", 'r') as f:
        infer_config = yaml.safe_load(f)
    data_root = os.path.dirname(train_config['dataset']['train_metadata'])
    # 1. Pre-processing: ensure processed data exists
    train_meta_path = train_config['dataset']['train_metadata']
    if not os.path.exists(train_meta_path):
        raise RuntimeError("Processed training metadata not found. Please run data preprocessing before pipeline.")
    # (Optional: call a preprocessing script here if needed, e.g., process.py to generate train_metadata.csv)
    # 2. Initial training (with no pseudo, maybe no synthetic)
    # Override config flags for initial run
    train_config['dataset']['include_pseudo'] = False
    train_config['dataset']['include_synthetic'] = False
    # Also possibly use a different learning rate or epochs for initial stage if desired
    initial_train_config_path = "config/_initial_train_tmp.yaml"
    with open(initial_train_config_path, 'w') as f:
        yaml.safe_dump(train_config, f)
    print("Starting initial training...")
    subprocess.run(["python", "train_efficientnet.py"], check=True)
    subprocess.run(["python", "train_regnety.py"], check=True)
    print("Initial training completed.")
    # After initial training, copy the best model weights as "initial" checkpoints for next loop (for fine-tuning).
    models_dir = infer_config['paths']['models_dir']
    eff_dir = os.path.join(models_dir, "efficientnet_b0")
    reg_dir = os.path.join(models_dir, "regnety_008")
    # Identify best checkpoints for each run (assuming train scripts print them or we pick last epoch of each run).
    # Here, for simplicity, assume the highest epoch files for run1, run2, run3 in each arch are best.
    initial_eff_ckpts = []
    initial_reg_ckpts = []
    for run in range(1, train_config['model']['architectures'][0]['num_models']+1):
        # EfficientNet
        ckpt_files = [f for f in os.listdir(eff_dir) if f.startswith("efficientnet_b0_run%d" % run)]
        if not ckpt_files:
            continue
        best_ckpt = sorted(ckpt_files, key=lambda x: int(x.split('_epoch')[-1].split('.pth')[0]))[-1]  # highest epoch
        src = os.path.join(eff_dir, best_ckpt)
        dst = os.path.join(eff_dir, f"efficientnet_b0_initial_{run}.pth")
        os.replace(src, dst)
        initial_eff_ckpts.append(dst)
        # RegNet
        ckpt_files = [f for f in os.listdir(reg_dir) if f.startswith("regnety_008_run%d" % run)]
        if not ckpt_files:
            continue
        best_ckpt = sorted(ckpt_files, key=lambda x: int(x.split('_epoch')[-1].split('.pth')[0]))[-1]
        src = os.path.join(reg_dir, best_ckpt)
        dst = os.path.join(reg_dir, f"regnety_008_initial_{run}.pth")
        os.replace(src, dst)
        initial_reg_ckpts.append(dst)
    # Update config to use these as init_checkpoint for fine-tuning
    for arch in train_config['model']['architectures']:
        if arch['name'].startswith("efficientnet"):
            arch['init_checkpoint'] = os.path.join(models_dir, "efficientnet_b0", "efficientnet_b0_initial")
        elif arch['name'].startswith("regnety"):
            arch['init_checkpoint'] = os.path.join(models_dir, "regnety_008", "regnety_008_initial")
    train_config['dataset']['include_pseudo'] = True
    train_config['dataset']['include_synthetic'] = True  # now include synthetic as well
    # Save updated config for retraining
    retrain_config_path = "config/_retrain_tmp.yaml"
    with open(retrain_config_path, 'w') as f:
        yaml.safe_dump(train_config, f)
    # 3. Iterative training loops with pseudo-label refinement
    max_loops = train_config.get('automl_loops', 5)
    for loop in range(1, max_loops):
        print(f"Starting training loop {loop} with pseudo-labels...")
        # Train models (fine-tune from initial weights including pseudo and synthetic)
        subprocess.run(["python", "train_efficientnet.py"], check=True)
        subprocess.run(["python", "train_regnety.py"], check=True)
        print(f"Loop {loop} training completed. Generating pseudo-labels for next loop...")
        # 4. Inference on training soundscapes to update pseudo-labels
        # Load the newly trained ensemble (all 6 models) for inference on unlabeled data
        ensemble_ckpts = infer_config['ensemble']['checkpoints']  # expecting final names (which we might not have yet)
        # If not explicitly set, we can derive from last training run outputs.
        # For simplicity, use initial checkpoint names (which now hold updated weights after fine-tuning)
        ensemble_ckpts = []
        for run in range(1, train_config['model']['architectures'][0]['num_models']+1):
            ensemble_ckpts.append(f"efficientnet_b0_initial_{run}.pth")
        for run in range(1, train_config['model']['architectures'][1]['num_models']+1):
            ensemble_ckpts.append(f"regnety_008_initial_{run}.pth")
        # Load models
        models = []
        for ckpt_name in ensemble_ckpts:
            if ckpt_name.startswith("efficientnet"):
                arch = "efficientnet_b0"
            else:
                arch = "regnety_008"
            ckpt_path = os.path.join(models_dir, arch, ckpt_name)
            model, species_map = load_model(arch, len(species_map) if species_map else len(class_map), ckpt_path, torch.device("cpu"))
            models.append(model)
        # Run inference on unlabeled soundscapes (assuming they are in data_root/train_soundscapes)
        unlabeled_dir = os.path.join(data_root, "train_soundscapes")
        if not os.path.isdir(unlabeled_dir):
            print("No unlabeled soundscapes directory found; skipping pseudo-label generation.")
            break
        pseudo_entries = []
        for fname in os.listdir(unlabeled_dir):
            if not fname.lower().endswith(('.wav', '.ogg', '.mp3')):
                continue
            file_path = os.path.join(unlabeled_dir, fname)
            y, sr = librosa.load(file_path, sr=None)
            # Obtain ensemble predictions for this file's 5s chunks
            all_preds = []
            for model in models:
                preds = predict_chunks(model, y, sr, chunk_duration=5.0)
                all_preds.append(preds)
            # Smooth and ensemble
            for m in range(len(all_preds)):
                all_preds[m] = smooth_predictions(all_preds[m], smoothing_window=5)
            combined = ensemble_predictions(all_preds, strategy='min_then_avg')
            # Generate pseudo-label dictionary for each chunk above threshold
            species_list = sorted(species_map.keys()) if species_map else sorted(class_map.keys())
            pseudo_labels = create_pseudo_labels(combined, species_list, threshold=presence_threshold)
            # Build DataFrame rows for these pseudo-labels
            file_id = os.path.splitext(fname)[0]
            for i, label_dict in enumerate(pseudo_labels):
                if not label_dict:
                    # If no species above threshold in this chunk, we can optionally include it as all-negative example
                    # but we might skip adding it to avoid overwhelming with negatives.
                    continue
                row = {
                    "filename": file_id,
                    "label_json": json.dumps(label_dict),
                    "weight": 0.5  # pseudo-label weight
                }
                pseudo_entries.append(row)
        # Save pseudo labels to CSV
        if pseudo_entries:
            pseudo_df = pd.DataFrame(pseudo_entries)
            pseudo_csv_path = os.path.join(data_root, "processed", "soundscape_metadata.csv")
            pseudo_df.to_csv(pseudo_csv_path, index=False)
            print(f"Pseudo-labels updated: {len(pseudo_entries)} chunks labeled and saved to {pseudo_csv_path}")
        else:
            print("No pseudo-labels generated in this iteration.")
    # 5. Final inference on test soundscapes
    print("Final training complete. Running inference on test set...")
    subprocess.run(["python", "inference.py"], check=True)
    print("AutoML pipeline finished. Submission file is ready.")

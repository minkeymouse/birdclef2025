# train_efficientnet.py
import yaml
import json
import torch
from train_utils import BirdClefDataset, create_dataloader, train_model

if __name__ == "__main__":
    # Load training config
    with open("config/train.yaml", 'r') as f:
        config = yaml.safe_load(f)
    # Identify EfficientNet config
    arch_config = None
    for arch in config['model']['architectures']:
        if arch['name'].startswith("efficientnet"):
            arch_config = arch
            break
    if arch_config is None:
        raise ValueError("EfficientNet configuration not found in train.yaml")
    model_name = arch_config['name']
    num_models = arch_config.get('num_models', 1)
    pretrained = arch_config.get('pretrained', True)
    # Load training metadata
    train_meta_path = config['dataset']['train_metadata']
    df = pd.read_csv(train_meta_path)
    # If pseudo-labeled samples should be included and exist, append them
    if config['dataset'].get('include_pseudo', False):
        pseudo_meta_path = os.path.join(os.path.dirname(train_meta_path), "soundscape_metadata.csv")
        if os.path.exists(pseudo_meta_path):
            df_pseudo = pd.read_csv(pseudo_meta_path)
            # We assume pseudo metadata already in same format (including label_json and weight)
            df = pd.concat([df, df_pseudo], ignore_index=True)
    # If synthetic samples should be included, they are assumed to be already in train_metadata or processed.
    # (If synthetic are separate, one could similarly read and append them here.)
    # Create class map (species to index)
    if 'label_json' in df.columns:
        # Build class set from all labels present
        species_set = {sp for js in df['label_json'] for sp in json.loads(js).keys()}
    else:
        # If no label_json, assume a 'primary_label' column exists
        species_set = set(df['primary_label'].unique())
        # Also include any secondary labels if provided
        if 'secondary_labels' in df.columns:
            for sec_list in df['secondary_labels']:
                if isinstance(sec_list, str):
                    for sp in sec_list.split():
                        species_set.add(sp)
    species_list = sorted(species_set)
    class_map = {sp: idx for idx, sp in enumerate(species_list)}
    num_classes = len(class_map)
    # Split train/val (for simplicity, use a small percentage as validation)
    val_frac = config['training'].get('val_fraction', 0.1)
    df_val = df.sample(frac=val_frac, random_state=42)
    df_train = df.drop(df_val.index).reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    # Create datasets
    train_dataset = BirdClefDataset(df_train, class_map, mel_shape=(128, 256), augment=True)
    val_dataset = BirdClefDataset(df_val, class_map, mel_shape=(128, 256), augment=False)
    train_loader = create_dataloader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train specified number of models (e.g., 3 seeds for ensemble)
    saved_checkpoints = []
    base_seed = config['training'].get('seed', 42)
    for run in range(1, num_models+1):
        # Set reproducibility seed
        torch.manual_seed(base_seed + run)
        np.random.seed(base_seed + run)
        # Initialize model
        from torchvision import models
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        # Adjust final layer for num_classes
        num_feats = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(num_feats, num_classes)
        model = model.to(device)
        # If fine-tuning from a provided initial checkpoint (for pseudo label iterations)
        init_ckpt_base = arch_config.get('init_checkpoint', None)
        if init_ckpt_base:
            ckpt_path = f"{init_ckpt_base}_{run}.pth"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded initial checkpoint for {model_name} run{run} from {ckpt_path}")
        # Prepare config for train_model
        run_config = config.copy()
        run_config['current_arch'] = model_name
        run_config['current_run'] = run
        run_config['class_map'] = class_map
        run_config['paths'] = {
            'models_dir': os.path.join("models", model_name)  # save in models/<architecture> folder
        }
        os.makedirs(run_config['paths']['models_dir'], exist_ok=True)
        # Train model
        best_ckpts = train_model(model, train_loader, val_loader, run_config, device)
        saved_checkpoints.extend(best_ckpts)
    # After training, output the list of saved checkpoint paths
    print("Training complete. Saved model checkpoints:")
    for ckpt in saved_checkpoints:
        print(" -", ckpt)


class Config:
 
    DEBUG_MODE = False
    
    OUTPUT_DIR = '/kaggle/working/'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    FS = 32000
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)  
    
    N_MAX = 50 if DEBUG_MODE else None  

config = Config()

print("Starting audio processing...")
print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")
start_time = time.time()

all_bird_data = {}
errors = []

for i, row in tqdm(working_df.iterrows(), total=total_samples):
    if config.N_MAX is not None and i >= config.N_MAX:
        break
    
    try:
        audio_data, _ = librosa.load(row.filepath, sr=config.FS)

        target_samples = int(config.TARGET_DURATION * config.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, 
                                 (0, target_samples - len(center_audio)), 
                                 mode='constant')

        mel_spec = audio2melspec(center_audio)

        if mel_spec.shape != config.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        all_bird_data[row.samplename] = mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {row.filepath}: {e}")
        errors.append((row.filepath, str(e)))

end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")
print(f"Successfully processed {len(all_bird_data)} files out of {total_samples} total")
print(f"Failed to process {len(errors)} files")











df = pd.read_csv(TRAIN_CSV)
golden_rating = sel_cfg["golden_rating"]
rare_species_thresh = sel_cfg["rare_species_threshold"]

best_rating_df = df[df["rating"] >= golden_rating]
label_counts = df.groupby("primary_label")["filename"].transform("count")
minority_df = df[label_counts < rare_species_thresh]
golden_df = pd.concat([best_rating_df, minority_df], ignore_index=True)
selected_labels = set(golden_df["primary_label"].unique())

for offset in [0.5, 1, 1.5, 2]:
    threshold = golden_rating - offset
    sub_df = df[df["rating"] >= threshold]
    new_df = sub_df[~sub_df["primary_label"].isin(selected_labels)]
    if not new_df.empty:
        golden_df = pd.concat([golden_df, new_df], ignore_index=True)
        selected_labels |= set(new_df["primary_label"].unique())

golden_df = golden_df.drop_duplicates(subset=["filename"])

chunk_sec = chunk_cfg["train_chunk_duration"]
hop_sec = chunk_cfg["train_chunk_hop"]
sample_rate = audio_cfg["sample_rate"]
chunk_samples = int(chunk_sec * sample_rate)
hop_samples = int(hop_sec * sample_rate)

meta_rows: List[dict] = []
for rec in golden_df.itertuples(index=False):
    fname = rec.filename
    label = str(rec.primary_label)
    audio_path = AUDIO_DIR / fname
    if not audio_path.exists():
        log.warning("Missing file: %s", fname)
        continue
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    h = hashlib.md5(y.tobytes()).hexdigest()
    if dedup_cfg.get("enabled", False) and h in seen_hashes:
        continue
    seen_hashes.add(h)
    if audio_cfg.get("trim_top_db") is not None:
        y, _ = librosa.effects.trim(y, top_db=audio_cfg["trim_top_db"])
    if y.size == 0:
        continue
    min_dur = int(audio_cfg["min_duration"] * sample_rate)
    if y.size < min_dur:
        reps = int(np.ceil(min_dur / y.size))
        y = np.tile(y, reps)[:min_dur]
    total = len(y)
    ptr = 0
    while ptr + chunk_samples <= total:
        chunk = y[ptr : ptr + chunk_samples]
        ptr += hop_samples
        if utils.is_silent(chunk, db_thresh=audio_cfg.get("silence_thresh_db", -50.0)):
            continue
        if utils.contains_voice(chunk, sample_rate):
            continue
        m = librosa.feature.melspectrogram(
            y=chunk,
            sr=sample_rate,
            n_fft=mel_cfg["n_fft"],
            hop_length=mel_cfg["hop_length"],
            n_mels=mel_cfg["n_mels"],
            fmin=mel_cfg["fmin"],
            fmax=mel_cfg["fmax"],
            power=mel_cfg["power"],
        )
        mel_db = librosa.power_to_db(m, ref=np.max)
        mel_db = utils.resize_mel(mel_db, *mel_cfg["target_shape"]).astype(np.float32)
        chunk_id = utils.hash_chunk_id(fname, ptr / sample_rate)
        mel_path = MEL_DIR / f"{chunk_id}.npy"
        label_path = LABEL_DIR / f"{chunk_id}.npy"
        np.save(mel_path, mel_db)
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        idx = class_map.get(label)
        if idx is None:
            continue
        lbl[idx] = 1.0
        np.save(label_path, lbl)
        meta_rows.append(
            {
                "filename": fname,
                "end_sec": round(ptr / sample_rate, 3),
                "mel_path": str(mel_path),
                "label_path": str(label_path),
                "weight": float(label_cfg.get("golden_label_weight", 1.0)),
            }
        )

meta_df = pd.DataFrame(meta_rows)
if METADATA_CSV.exists():
    existing = pd.read_csv(METADATA_CSV)
    meta_df = pd.concat([existing, meta_df], ignore_index=True)
meta_df.to_csv(METADATA_CSV, index=False)

with HASH_FILE.open("w") as f:
    f.write("\n".join(sorted(seen_hashes)))


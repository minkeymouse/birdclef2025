#!/usr/bin/env python3
import pandas as pd

# — PARAMETERS —————————————————————————————————————————————————————
TRAIN_CSV    = "data/birdclef/train.csv"       # path to your train.csv
META_CSV     = "data/birdclef/DATABASE/train_metadata.csv"  # path to your existing metadata
THRESHOLD    = 100   # species with fewer than 50 examples are “rare” (adjust as needed)
# — END PARAMETERS ——————————————————————————————————————————————————

# 1) Load train.csv and compute species frequencies
train_df = pd.read_csv(TRAIN_CSV)
counts   = train_df["primary_label"].value_counts()
print("Species counts:\n", counts)

# Identify rare species
rare_species = counts[counts < THRESHOLD].index.tolist()
print(f"\nFound {len(rare_species)} species with < {THRESHOLD} samples:\n", rare_species)

# 2) Load your metadata and bring in the primary_label for each chunk via filename
meta_df = pd.read_csv(META_CSV)

# Map filename → primary_label
label_map = train_df.set_index("filename")["primary_label"]
meta_df["primary_label"] = meta_df["filename"].map(label_map)

# 3) Update weights for rare-species chunks
mask = meta_df["primary_label"].isin(rare_species)
print(f"\nUpdating {mask.sum()} chunks to weight = 1.0")
meta_df.loc[mask, "weight"] = 1.0

# 4) (Optional) Drop the helper column before saving
meta_df = meta_df.drop(columns="primary_label")

# 5) Save out the updated metadata
meta_df.to_csv(META_CSV, index=False)
print(f"\nSaved updated metadata to {META_CSV}")

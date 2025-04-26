set -e
python src/process/process_update_labels.py
python src/train/train_efficientnet.py
python src/train/train_regnety.py
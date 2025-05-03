#!/usr/bin/env bash
set -e

#python src/process/process_mixup.py
for i in {1..50}; do
  python src/process/process_update_labels.py
done

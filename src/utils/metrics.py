# metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score

def macro_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged ROC-AUC, skipping classes with no true positives."""
    num_classes = y_true.shape[1]
    aucs = []
    for i in range(num_classes):
        if np.sum(y_true[:, i] == 1) == 0:
            continue  # skip classes with no positives in truth
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.0

def macro_precision_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Macro-average precision at given threshold (skip classes with no predictions and no truths)."""
    num_classes = y_true.shape[1]
    precisions = []
    for i in range(num_classes):
        pred_pos = y_pred[:, i] >= threshold
        true_pos = y_true[:, i] == 1
        if pred_pos.sum() == 0:
            if true_pos.sum() == 0:
                continue
            else:
                precisions.append(0.0)
        else:
            prec_i = np.sum(true_pos & pred_pos) / max(1, pred_pos.sum())
            precisions.append(prec_i)
    return float(np.mean(precisions)) if precisions else 0.0

def create_pseudo_labels(chunk_probs: np.ndarray, species_list: list, threshold: float = 0.5) -> list:
    """
    Given chunk-level probabilities (num_chunks x num_classes) and species list mapping indices to species codes,
    return a list of pseudo-label dicts for each chunk where probabilities exceed threshold.
    Each dict maps species code -> 1.0 for high-confidence presence.
    """
    num_chunks, num_classes = chunk_probs.shape
    pseudo_labels = []
    for t in range(num_chunks):
        label_dict = {}
        for j in range(num_classes):
            if chunk_probs[t, j] >= threshold:
                # Assign a soft label of 1.0 (presence) for that species in this chunk
                label_dict[species_list[j]] = 1.0
        pseudo_labels.append(label_dict)
    return pseudo_labels

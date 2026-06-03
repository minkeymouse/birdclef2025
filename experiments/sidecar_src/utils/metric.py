import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores: list[float] = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.max() <= 0 or col.min() >= 1:
            continue
        try:
            scores.append(float(roc_auc_score(col, y_pred[:, j])))
        except ValueError:
            continue
    if not scores:
        return float("nan")
    return float(np.mean(scores))


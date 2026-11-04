"""Per-emotion threshold tuning on dev only."""
import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds(y_true, proba, grid=np.arange(0.1, 0.91, 0.05)):
    thresholds = []
    for i in range(y_true.shape[1]):
        scores = np.array([f1_score(y_true[:, i], proba[:, i] >= t, zero_division=0)
                           for t in grid])
        best = grid[scores >= scores.max() - 1e-12]
        thresholds.append(float(best[np.argmin(np.abs(best - 0.5))]))  # ties -> closest to 0.5
    return np.array(thresholds)
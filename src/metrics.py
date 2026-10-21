"""Shared evaluation helpers."""
import numpy as np
from scipy.stats import rankdata


def roc_auc(scores, labels):
    """Global ROC AUC, ties counted as one half (same definition as the leaderboard)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)                 # average ranks for ties
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

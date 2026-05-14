"""Evaluation metrics for binary stress classification.

All metrics assume a binary problem:
    class 0 = Non-stressed (100% P dose, healthy)
    class 1 = Stressed     (25 / 50 / 75% P dose)

Primary metric: PR-AUC (Average Precision), chosen because of class
imbalance (~74% stressed in the test set).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the full set of evaluation metrics.

    Parameters
    ----------
    y_true : ndarray of shape (N,)
        Ground-truth binary labels (0 / 1).
    y_prob : ndarray of shape (N,)
        Predicted probabilities for class 1 (stressed).
    threshold : float
        Decision threshold applied to *y_prob* to produce hard labels.

    Returns
    -------
    dict with keys:
        PR-AUC, ROC-AUC, F1-macro, Recall_1, Precision_1, Accuracy
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "PR-AUC": round(float(average_precision_score(y_true, y_prob)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "F1-macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "Recall_1": round(float(recall_score(y_true, y_pred)), 4),
        "Precision_1": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }

"""Inference helpers for scikit-learn-compatible ML models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score


def predict_proba_ml(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from an sklearn pipeline.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model with a ``predict_proba`` method.
    X : ndarray of shape (N, n_features)
        Feature matrix in the *original* (unscaled) space; the pipeline
        includes the internal RobustScaler.

    Returns
    -------
    ndarray of shape (N,)
        P(y=1 | x) for each sample.
    """
    return model.predict_proba(X)[:, 1]


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_grid: int = 161,
) -> float:
    """Find the decision threshold that maximises macro-F1 on a validation set.

    Parameters
    ----------
    y_true : ndarray of shape (N,)
        Ground-truth binary labels.
    y_prob : ndarray of shape (N,)
        Predicted probabilities for class 1.
    n_grid : int, optional
        Number of threshold candidates in [0.10, 0.90].  Defaults to 161.

    Returns
    -------
    float
        Optimal threshold.
    """
    grid = np.linspace(0.10, 0.90, n_grid)
    return float(
        max(
            grid,
            key=lambda t: f1_score(y_true, (y_prob >= t).astype(int), average="macro"),
        )
    )

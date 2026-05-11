"""Inference-time feature ablation utilities.

Ablation neutralises selected feature positions by replacing their values with
the per-feature training mean.  No retraining is performed.

Functions
---------
ablate_features
    Replace selected feature columns with their training-set mean.
evaluate_ablation_grid
    Evaluate a model under multiple ablation conditions and return a results
    DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score,
)


def ablate_features(
    X: np.ndarray,
    feature_indices: list[int],
    replacement: str = "mean",
    train_stats: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of ``X`` with selected features neutralised.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to ablate.
    feature_indices : list of int
        Column indices to neutralise.
    replacement : {"mean"}
        Strategy for computing the replacement value.  Only ``"mean"`` is
        currently supported.
    train_stats : ndarray of shape (n_features,), optional
        Pre-computed per-feature replacement values (e.g. training-set means).
        If ``None``, the mean of ``X`` itself is used — which is typically
        wrong for test-set ablation; always pass training-set stats.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        Copy of ``X`` with the specified columns replaced.
    """
    if replacement != "mean":
        raise ValueError(f"Unsupported replacement strategy: '{replacement}'.")

    X_abl = X.copy()
    stats = train_stats if train_stats is not None else X.mean(axis=0)
    for idx in feature_indices:
        X_abl[:, idx] = stats[idx]
    return X_abl


def evaluate_ablation_grid(
    predict_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_groups: dict[str, list[int]],
    threshold: float,
    train_stats: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """Evaluate ``predict_fn`` under each ablation condition in ``feature_groups``.

    Parameters
    ----------
    predict_fn : callable
        Function ``f(X) -> y_prob`` returning predicted probabilities for the
        positive class.
    X_test : ndarray of shape (n_samples, n_features)
        Test feature matrix (pre-scaled for neural models).
    y_test : ndarray of shape (n_samples,)
        Ground-truth binary labels.
    feature_groups : dict mapping condition_name → list of feature indices
        Empty list means baseline (no ablation).
    threshold : float
        Decision threshold applied to probabilities to produce hard labels.
    train_stats : ndarray of shape (n_features,)
        Per-feature training-set means used as replacement values.
    model_name : str, optional
        Label added to the ``Modelo`` column.

    Returns
    -------
    pd.DataFrame
        Rows indexed by (model_name, condition_name) with metric columns:
        PR-AUC, ROC-AUC, F1-macro, Recall_1, Precision_1, Accuracy.
    """
    rows = []
    for cond_name, indices in feature_groups.items():
        if indices:
            X_abl = ablate_features(X_test, indices, train_stats=train_stats)
        else:
            X_abl = X_test

        y_prob = predict_fn(X_abl)
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "Modelo":      model_name,
            "Condicion":   cond_name,
            "PR-AUC":      round(average_precision_score(y_test, y_prob), 4),
            "ROC-AUC":     round(roc_auc_score(y_test, y_prob), 4),
            "F1-macro":    round(f1_score(y_test, y_pred, average="macro"), 4),
            "Recall_1":    round(recall_score(y_test, y_pred), 4),
            "Precision_1": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        })

    return pd.DataFrame(rows)

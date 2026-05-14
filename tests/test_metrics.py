"""Smoke tests for evaluation metrics on synthetic data."""

import numpy as np

from spectralcrop.evaluation.metrics import compute_all_metrics
from spectralcrop.models.ml.predict import find_best_threshold


def test_compute_all_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    thr = 0.5
    m = compute_all_metrics(y_true, y_prob, thr)
    assert set(m.keys()) == {"PR-AUC", "ROC-AUC", "F1-macro", "Recall_1", "Precision_1", "Accuracy"}
    assert m["PR-AUC"] == 1.0
    assert m["ROC-AUC"] == 1.0
    assert m["Accuracy"] == 1.0


def test_compute_all_metrics_random():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(size=200)
    m = compute_all_metrics(y_true, y_prob, threshold=0.5)
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_find_best_threshold_binary():
    # Perfect separation → optimal threshold somewhere between 0.5 and 0.9
    rng = np.random.default_rng(7)
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
    thr = find_best_threshold(y_true, y_prob)
    assert 0.10 <= thr <= 0.90, f"Threshold out of grid range: {thr}"
    # With near-perfect separation, F1 should be high at the chosen threshold
    from sklearn.metrics import f1_score

    f1 = f1_score(y_true, (y_prob >= thr).astype(int), average="macro")
    assert f1 > 0.9, f"Expected high F1 with perfect separation, got {f1:.3f}"

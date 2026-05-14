"""Confusion matrix utilities for binary classification evaluation.

Functions
---------
compute_normalized_confusion_matrices
    Compute absolute and three normalized confusion matrix variants.
plot_confusion_matrix_grid
    Plot a 3×2 grid of row-normalized confusion matrices with shared axes.
describe_evaluable_pixels_2d
    Summarize the pixel-count reduction introduced by CNN-2D patch geometry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_normalized_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return absolute and three normalized confusion matrices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 / 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 / 1).

    Returns
    -------
    dict with keys:
        ``abs``  – raw count matrix (2×2 ndarray, int)
        ``row``  – row-normalized matrix (recall per class, %)
        ``col``  – column-normalized matrix (precision per class, %)
        ``total``– total-normalized matrix (% of all samples)
    """
    cm = confusion_matrix(y_true, y_pred)
    return {
        "abs": cm,
        "row": cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100,
        "col": cm.astype(float) / cm.sum(axis=0, keepdims=True) * 100,
        "total": cm.astype(float) / cm.sum() * 100,
    }


def plot_confusion_matrix_grid(
    matrices_dict: dict,
    n_eval: dict,
    output_path: str | Path,
    class_labels: list[str] | None = None,
    figsize: tuple[float, float] = (10, 14),
    dpi: int = 300,
) -> None:
    """Plot a 3×2 grid of row-normalized confusion matrices and save to disk.

    The x-axis (predicted class) is shared on the bottom row only; the y-axis
    (true class) is shared on the left column only.  A single colorbar is
    placed to the right of the right column.

    Parameters
    ----------
    matrices_dict : dict
        Mapping ``model_name → {"abs": cm_abs, "row": cm_row, ...}`` as
        returned by :func:`compute_normalized_confusion_matrices`.
    n_eval : dict
        Mapping ``model_name → int`` with the number of evaluated samples.
    output_path : str or Path
        Directory where ``confusion_matrices_pct.{png,pdf}`` will be saved.
    class_labels : list of str, optional
        Labels for the two classes.  Defaults to ``["Sano (0)", "Estresado (1)"]``.
    figsize : tuple, optional
        Figure size in inches.  Defaults to ``(10, 14)``.
    dpi : int, optional
        Resolution for raster output.  Defaults to ``300``.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if class_labels is None:
        class_labels = ["Sano (0)", "Estresado (1)"]

    model_names = list(matrices_dict.keys())
    if len(model_names) != 6:
        raise ValueError(f"Expected exactly 6 models, got {len(model_names)}.")

    model_grid = [model_names[i : i + 2] for i in range(0, 6, 2)]

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    for row_idx, row_models in enumerate(model_grid):
        for col_idx, m in enumerate(row_models):
            ax = axes[row_idx, col_idx]
            cm_abs = matrices_dict[m]["abs"]
            cm_pct = matrices_dict[m]["row"]

            annot = np.empty((2, 2), dtype=object)
            for r in range(2):
                for c in range(2):
                    annot[r, c] = f"{cm_pct[r, c]:.2f}%\n({cm_abs[r, c]:,})"

            show_x = row_idx == 2
            show_y = col_idx == 0

            sns.heatmap(
                cm_pct,
                ax=ax,
                annot=annot,
                fmt="",
                cmap="Blues",
                vmin=0,
                vmax=100,
                xticklabels=class_labels if show_x else False,
                yticklabels=class_labels if show_y else False,
                linewidths=0.5,
                linecolor="gray",
                cbar=False,
                annot_kws={"size": 9},
            )

            ax.set_title(f"{m}  (n = {n_eval[m]:,})", fontsize=10, fontweight="bold")
            ax.set_ylabel("Real" if col_idx == 0 else "", fontsize=9)
            ax.set_xlabel("Predicho" if row_idx == 2 else "", fontsize=9)

            if show_y:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            if show_x:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = mpl.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 1], orientation="vertical", shrink=0.9, pad=0.03, aspect=30)
    cbar.set_label("% (normalizado por fila)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Matrices de confusión normalizadas por fila (recall por clase) — conjunto de prueba\n"
        "Cada fila suma 100 %. Entre paréntesis: conteo absoluto.",
        fontsize=11,
        fontweight="bold",
    )

    for ext in ("png", "pdf"):
        fig.savefig(output_path / f"confusion_matrices_pct.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close()


def describe_evaluable_pixels_2d(
    n_ml: int,
    n_2d: int,
    patch_size: int = 5,
) -> pd.DataFrame:
    """Build a table summarising the pixel-count reduction in CNN-2D evaluation.

    Parameters
    ----------
    n_ml : int
        Number of pixels evaluated by ML / CNN-1D models (flat vector).
    n_2d : int
        Number of pixels evaluated by CNN-2D (patch-based).
    patch_size : int, optional
        Spatial patch size used by CNN-2D.  Defaults to ``5``.

    Returns
    -------
    pd.DataFrame
        One row per architecture with columns:
        ``Arquitectura``, ``Pixeles evaluados``, ``Pct vs ML``, ``Razon``.
    """
    pct = n_2d / n_ml * 100
    r = patch_size // 2
    return pd.DataFrame(
        [
            {
                "Arquitectura": "ML (LR / SGD / LightGBM / XGBoost)",
                "Pixeles evaluados": n_ml,
                "Pct vs ML": "100.00%",
                "Razon": "Vector plano — todos los píxeles válidos del split de prueba",
            },
            {
                "Arquitectura": "CNN-1D",
                "Pixeles evaluados": n_ml,
                "Pct vs ML": "100.00%",
                "Razon": "Vector plano — todos los píxeles válidos del split de prueba",
            },
            {
                "Arquitectura": "CNN-2D",
                "Pixeles evaluados": n_2d,
                "Pct vs ML": f"{pct:.2f}%",
                "Razon": (
                    f"Parches {patch_size}×{patch_size}: excluye ~{r} px del borde de "
                    "cada parcela y píxeles con NaN en la vecindad"
                ),
            },
        ]
    )

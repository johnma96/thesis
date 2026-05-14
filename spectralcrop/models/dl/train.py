"""Training loop for DL models with early stopping.

This module implements the training procedure used in
notebooks/305-jmmz-dl-binary-modeling.ipynb.  The CNN-2D final
hyperparameters are locked in spectralcrop/config/constants.py.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_prauc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute PR-AUC on a validation DataLoader."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    return float(average_precision_score(all_labels, all_probs))


def fit_cnn2d(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
    seed: int = 42,
    on_epoch_end: Callable[[int, float, float], None] | None = None,
) -> tuple[nn.Module, list[float], list[float]]:
    """Train CNN-2D with early stopping on val PR-AUC.

    Parameters
    ----------
    model : nn.Module
        Uninitialised (or newly instantiated) CNN-2D.
    X_train, y_train : ndarray
        Training patches ``(N, C, H, W)`` and labels.
    X_val, y_val : ndarray
        Validation patches and labels.
    lr : float
        Adam learning rate.
    batch_size : int
    max_epochs : int
    patience : int
        Number of epochs without improvement before stopping.
    weight_decay : float
        Adam weight decay (L2 regularisation).
    device : torch.device
    seed : int
        For DataLoader worker reproducibility.
    on_epoch_end : callable, optional
        Hook called at the end of each epoch with
        ``(epoch, train_loss, val_prauc)``.

    Returns
    -------
    model : nn.Module
        Best model (weights restored to best val PR-AUC epoch).
    train_losses : list of float
    val_praucs : list of float
    """
    torch.manual_seed(seed)

    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_prauc = -1.0
    best_state = None
    no_improve = 0
    train_losses: list[float] = []
    val_praucs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_prauc = eval_prauc(model, val_loader, device)

        train_losses.append(tr_loss)
        val_praucs.append(vl_prauc)

        if on_epoch_end is not None:
            on_epoch_end(epoch, tr_loss, vl_prauc)

        if vl_prauc > best_prauc:
            best_prauc = vl_prauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best val PR-AUC=%.4f)", epoch, best_prauc)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_praucs

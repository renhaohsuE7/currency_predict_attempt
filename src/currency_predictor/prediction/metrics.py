"""Forecast evaluation metrics.

Standalone functions for computing MASE and MDA — usable from both
CurrencyPredictor and ModelComparer without circular imports.
"""

import numpy as np


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Mean Absolute Scaled Error.

    MASE = MAE(model) / MAE(naive_on_train)

    where naive_on_train = mean(|y_train[t] - y_train[t-1]|).

    * MASE < 1.0 → model beats the naive baseline
    * MASE = 1.0 → model is as good as naive
    * MASE > 1.0 → model is worse than naive

    Args:
        y_true: Actual values (test set).
        y_pred: Predicted values (same length as y_true).
        y_train: Training set values (used to compute naive scaling factor).

    Returns:
        MASE value.  Returns 0.0 if naive MAE is zero (constant training series).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mae_model = float(np.mean(np.abs(y_true - y_pred)))

    naive_errors = np.abs(np.diff(y_train))
    if len(naive_errors) == 0:
        return 0.0
    mae_naive = float(np.mean(naive_errors))

    if mae_naive == 0.0:
        return 0.0

    return mae_model / mae_naive


def mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Directional Accuracy.

    Fraction of correctly predicted price-change directions.

    * MDA > 0.5 → better than random
    * MDA = 1.0 → perfect directional prediction

    Args:
        y_true: Actual price series (length N).
        y_pred: Predicted price series (length N).

    Returns:
        Directional accuracy in [0.0, 1.0].  Returns 0.0 if fewer than 2 values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    min_len = min(len(true_dir), len(pred_dir))
    if min_len == 0:
        return 0.0

    return float(np.mean(true_dir[:min_len] == pred_dir[:min_len]))

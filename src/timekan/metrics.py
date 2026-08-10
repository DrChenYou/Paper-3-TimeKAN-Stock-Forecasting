"""Forecast regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(targets, predictions) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have matching shapes")
    nonzero = np.abs(targets) > 1e-12
    mape = np.mean(np.abs((targets[nonzero] - predictions[nonzero]) / targets[nonzero])) * 100
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "mape_percent": float(mape),
        "r2": float(r2_score(targets, predictions)),
    }

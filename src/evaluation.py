"""
Model evaluation utilities and comparison helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Return Mean Absolute Percentage Error (in percent)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute a standard set of regression metrics for a forecasting model.

    Parameters
    ----------
    name:
        Model name (used as a label in the returned dict).
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values.

    Returns
    -------
    dict
        Keys: ``Model``, ``MAE``, ``RMSE``, ``MAPE``.
    """
    return {
        "Model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "MAPE (%)": round(mean_absolute_percentage_error(y_true, y_pred), 4),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Build a sorted comparison table from a list of metric dicts.

    Parameters
    ----------
    results:
        Each element is the output of :func:`evaluate_model`.

    Returns
    -------
    pd.DataFrame
        Rows are models; sorted ascending by RMSE.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("RMSE").reset_index(drop=True)
    return df

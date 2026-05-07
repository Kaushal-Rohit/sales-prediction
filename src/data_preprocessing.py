"""
Data loading and preprocessing utilities for the sales-prediction project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV with ``ds`` (date) and ``y`` (target) columns.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a proper ``datetime64`` ``ds`` column.
    """
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def train_test_split_ts(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into train and test sets chronologically.

    Parameters
    ----------
    df:
        Full DataFrame sorted by date.
    test_ratio:
        Fraction of data to reserve for testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``
    """
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def create_sequences(
    series: np.ndarray, look_back: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D time series into supervised (X, y) pairs.

    Parameters
    ----------
    series:
        Scaled 1-D array of values.
    look_back:
        Number of past time steps used as input features.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` where ``X.shape == (samples, look_back, 1)``
        and ``y.shape == (samples,)``.
    """
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back : i])
        y.append(series[i])
    return np.array(X)[..., np.newaxis], np.array(y)


def scale_series(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit a ``MinMaxScaler`` on the training split and apply it to both splits.

    Parameters
    ----------
    train:
        1-D training values.
    test:
        1-D test values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, MinMaxScaler]
        ``(scaled_train, scaled_test, fitted_scaler)``
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test.reshape(-1, 1)).flatten()
    return scaled_train, scaled_test, scaler

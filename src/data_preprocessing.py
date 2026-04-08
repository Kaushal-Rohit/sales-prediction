"""
data_preprocessing.py
---------------------
Utility functions for loading and preparing the Superstore sales dataset
for time series forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV and return a cleaned DataFrame with a proper DatetimeIndex.

    Parameters
    ----------
    filepath : str
        Path to train.csv

    Returns
    -------
    pd.DataFrame
        Columns: ['Order Date', 'Sales', 'Category', 'Region', ...]
    """
    df = pd.read_csv(filepath)

    # Parse dates — handle multiple common formats
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Order Date', 'Sales'], inplace=True)

    # Remove negative sales (returns / data errors)
    df = df[df['Sales'] > 0].copy()

    return df


def aggregate_monthly(df: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    """
    Aggregate daily order data to monthly totals.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from load_and_clean()
    group_cols : list, optional
        Additional grouping columns e.g. ['Category', 'Region']

    Returns
    -------
    pd.DataFrame
        Monthly aggregated sales with columns ['ds', 'y'] (Prophet convention)
        or ['Date', 'Sales'] for LSTM
    """
    df = df.copy()
    df['Month'] = df['Order Date'].dt.to_period('M')

    if group_cols:
        monthly = df.groupby(['Month'] + group_cols)['Sales'].sum().reset_index()
    else:
        monthly = df.groupby('Month')['Sales'].sum().reset_index()

    monthly['Month'] = monthly['Month'].dt.to_timestamp()
    monthly.sort_values('Month', inplace=True)
    monthly.reset_index(drop=True, inplace=True)

    # Prophet-compatible column names
    monthly.rename(columns={'Month': 'ds', 'Sales': 'y'}, inplace=True)

    return monthly


def train_test_split_ts(df: pd.DataFrame, test_months: int = 6) -> tuple:
    """
    Split a monthly time series into train and test sets.
    Always keeps temporal order — no shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly DataFrame with 'ds' and 'y' columns
    test_months : int
        Number of trailing months to hold out as test set

    Returns
    -------
    tuple : (train_df, test_df)
    """
    split_idx = len(df) - test_months
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def create_lstm_sequences(series: np.ndarray, lookback: int = 12) -> tuple:
    """
    Convert a 1D time series into supervised (X, y) pairs for LSTM.

    Parameters
    ----------
    series : np.ndarray
        1D array of scaled sales values
    lookback : int
        Number of past time steps to use as input features

    Returns
    -------
    tuple : (X, y) where X.shape = (n, lookback, 1)
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y


def scale_series(train_values: np.ndarray, test_values: np.ndarray) -> tuple:
    """
    Fit MinMaxScaler on train, transform both train and test.
    Returns scaled arrays and the fitted scaler (needed for inverse transform).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
    return train_scaled, test_scaled, scaler

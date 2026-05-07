"""
Prophet-based time-series forecasting wrapper.

Prophet is an additive regression model developed by Facebook / Meta that
decomposes time-series into trend, seasonality and holiday components.  It
is highly interpretable and handles missing data / outliers gracefully but
may not capture complex non-linear patterns as well as deep-learning models.
"""

from __future__ import annotations

import pandas as pd
from prophet import Prophet  # type: ignore[import]


def build_prophet_model(
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    seasonality_mode: str = "additive",
) -> Prophet:
    """Instantiate and return a configured Prophet model.

    Parameters
    ----------
    yearly_seasonality:
        Include yearly seasonality.
    weekly_seasonality:
        Include weekly seasonality.
    daily_seasonality:
        Include daily seasonality.
    seasonality_mode:
        ``'additive'`` or ``'multiplicative'``.

    Returns
    -------
    Prophet
        An un-fitted Prophet instance.
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode,
    )
    return model


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    """Fit a Prophet model on the training data.

    Parameters
    ----------
    train_df:
        DataFrame with ``ds`` (datetime) and ``y`` (target) columns.

    Returns
    -------
    Prophet
        A fitted Prophet model.
    """
    model = build_prophet_model()
    model.fit(train_df[["ds", "y"]])
    return model


def predict_prophet(model: Prophet, future_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions from a fitted Prophet model.

    Parameters
    ----------
    model:
        A fitted Prophet model.
    future_df:
        DataFrame produced by ``model.make_future_dataframe()`` or any
        DataFrame with a ``ds`` column.

    Returns
    -------
    pd.DataFrame
        Prophet forecast DataFrame (contains ``yhat``, ``yhat_lower``,
        ``yhat_upper`` among other columns).
    """
    return model.predict(future_df)

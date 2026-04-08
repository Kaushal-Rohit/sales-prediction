"""
prophet_model.py
----------------
Training, forecasting, and plotting with Facebook Prophet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


def train_prophet(train_df: pd.DataFrame,
                  yearly_seasonality: bool = True,
                  weekly_seasonality: bool = False,
                  monthly_seasonality: bool = True,
                  changepoint_prior_scale: float = 0.1) -> Prophet:
    """
    Fit a Prophet model on monthly sales data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must have columns 'ds' (datetime) and 'y' (sales values)
    yearly_seasonality : bool
        Add a yearly Fourier seasonality component
    monthly_seasonality : bool
        Add a custom monthly seasonality component
    changepoint_prior_scale : float
        Flexibility of trend — higher = more flexible (risk of overfitting)

    Returns
    -------
    Fitted Prophet model
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode='multiplicative',   # better for sales data with growth
    )

    if monthly_seasonality:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    model.fit(train_df)
    print("[Prophet] Model fitted successfully.")
    return model


def forecast_prophet(model: Prophet, periods: int = 6,
                     freq: str = 'MS') -> pd.DataFrame:
    """
    Generate future forecasts with Prophet.

    Parameters
    ----------
    model : fitted Prophet model
    periods : int
        Number of future periods to predict
    freq : str
        Pandas frequency string — 'MS' = month start

    Returns
    -------
    pd.DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def evaluate_on_test(forecast: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Extract Prophet predictions that align with test set dates.

    Returns
    -------
    np.ndarray of predicted values matching test_df['ds']
    """
    forecast_subset = forecast[forecast['ds'].isin(test_df['ds'])][['ds', 'yhat']]
    merged = test_df.merge(forecast_subset, on='ds', how='left')
    return merged['yhat'].values


def plot_prophet_forecast(model: Prophet, forecast: pd.DataFrame,
                          train_df: pd.DataFrame, test_df: pd.DataFrame,
                          save_path: str = None):
    """
    Plot Prophet's in-sample fit + out-of-sample forecast with test actuals overlay.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # Training actuals
    ax.plot(train_df['ds'], train_df['y'], color='black', label='Training Actuals',
            linewidth=2, marker='o', markersize=4)

    # Test actuals
    ax.plot(test_df['ds'], test_df['y'], color='green', label='Test Actuals',
            linewidth=2, marker='o', markersize=5)

    # Forecast line
    ax.plot(forecast['ds'], forecast['yhat'], color='#4C72B0',
            label='Prophet Forecast', linewidth=2, linestyle='--')

    # Confidence interval
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    alpha=0.2, color='#4C72B0', label='95% Confidence Interval')

    # Mark the train/test split
    split_date = test_df['ds'].iloc[0]
    ax.axvline(split_date, color='red', linestyle=':', linewidth=1.5, label='Train/Test Split')

    ax.set_title("Prophet — Sales Forecast", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Monthly Sales ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()


def plot_prophet_components(model: Prophet, forecast: pd.DataFrame,
                            save_path: str = None):
    """Plot Prophet's decomposed trend + seasonality components."""
    fig = model.plot_components(forecast)
    fig.suptitle("Prophet — Decomposed Components", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()

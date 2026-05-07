"""
Visualisation helpers for sales-prediction results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_forecast_comparison(
    dates: pd.Series,
    actual: np.ndarray,
    prophet_pred: np.ndarray,
    lstm_pred: np.ndarray,
    output_path: str = "outputs/forecast_comparison.png",
) -> None:
    """Plot actual vs. Prophet vs. LSTM forecasts and save to file.

    Parameters
    ----------
    dates:
        Date series aligned with the test set (after ``look_back`` offset for
        LSTM is accounted for – caller is responsible for alignment).
    actual:
        Ground-truth sales values.
    prophet_pred:
        Prophet point forecasts.
    lstm_pred:
        LSTM point forecasts.
    output_path:
        File path where the figure is saved.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, actual, label="Actual", color="black", linewidth=1.5)
    ax.plot(dates, prophet_pred, label="Prophet", color="royalblue", linestyle="--")
    ax.plot(dates, lstm_pred, label="LSTM", color="tomato", linestyle="-.")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_title("Sales Forecast: Actual vs Prophet vs LSTM", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved forecast comparison plot → {output_path}")


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/metrics_comparison.png",
) -> None:
    """Bar-chart comparing MAE, RMSE, MAPE across models.

    Parameters
    ----------
    metrics_df:
        Output of :func:`src.evaluation.compare_models`.
    output_path:
        File path where the figure is saved.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metric_cols = ["MAE", "RMSE", "MAPE (%)"]
    x = np.arange(len(metrics_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, metrics_df[col], width, label=col)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_df["Model"])
    ax.set_title("Model Comparison – Error Metrics", fontsize=14)
    ax.set_ylabel("Error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics comparison plot → {output_path}")


def plot_prophet_components(
    model,
    forecast: pd.DataFrame,
    output_path: str = "outputs/prophet_components.png",
) -> None:
    """Save Prophet's built-in component plot.

    Parameters
    ----------
    model:
        Fitted Prophet model.
    forecast:
        Forecast DataFrame returned by ``model.predict()``.
    output_path:
        File path where the figure is saved.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig = model.plot_components(forecast)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Prophet components plot → {output_path}")


def plot_training_history(
    history,
    output_path: str = "outputs/lstm_training_history.png",
) -> None:
    """Plot LSTM training and validation loss curves.

    Parameters
    ----------
    history:
        ``keras.callbacks.History`` object returned by ``model.fit()``.
    output_path:
        File path where the figure is saved.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("LSTM Training History", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved LSTM training history plot → {output_path}")

"""
evaluate.py
-----------
Shared evaluation metrics for comparing time series forecasting models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(actual - predicted))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def evaluate_model(name: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE for a model and print a formatted summary.

    Returns
    -------
    dict with keys: model, MAE, RMSE, MAPE
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)

    print(f"\n{'='*40}")
    print(f"  {name} — Evaluation Results")
    print(f"{'='*40}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'='*40}\n")

    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_comparison(results: list, save_path: str = None):
    """
    Bar chart comparing MAE, RMSE, MAPE across models.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()
    save_path : str, optional — path to save PNG
    """
    df = pd.DataFrame(results)
    metrics = ["MAE", "RMSE", "MAPE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison: Prophet vs LSTM", fontsize=15, fontweight='bold')

    colors = ['#4C72B0', '#DD8452']

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df["model"], df[metric], color=colors, edgecolor='white', linewidth=1.2, width=0.5)
        ax.set_title(metric, fontsize=13)
        ax.set_ylabel(metric, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()


def plot_actual_vs_predicted(dates, actual, predicted_dict: dict,
                              title: str = "Actual vs Predicted Sales",
                              save_path: str = None):
    """
    Overlay actual and multiple model predictions on one chart.

    Parameters
    ----------
    dates : array-like of datetime
    actual : array-like of float
    predicted_dict : {'ModelName': predictions_array, ...}
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, actual, label='Actual', color='black', linewidth=2, marker='o', markersize=5)

    palette = ['#4C72B0', '#DD8452', '#55A868']
    for (name, pred), color in zip(predicted_dict.items(), palette):
        ax.plot(dates, pred, label=name, color=color, linewidth=2, linestyle='--', marker='s', markersize=4)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sales ($)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()

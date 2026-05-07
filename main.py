"""
End-to-end sales forecasting pipeline.

Runs the following steps:
  1. Generate (or load) synthetic retail sales data.
  2. Split into train / test sets.
  3. Fit a Prophet model and generate test-set forecasts.
  4. Build, train, and evaluate an LSTM model.
  5. Compare models using MAE, RMSE and MAPE.
  6. Save forecast and metric plots to the ``outputs/`` directory.

Usage
-----
    python main.py

All outputs (plots + a metrics CSV) are written to the ``outputs/`` folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress noisy TensorFlow / Keras logs before importing them
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data import generate_sales_data
from src.data_preprocessing import (
    train_test_split_ts,
    create_sequences,
    scale_series,
)
from src.prophet_model import fit_prophet, predict_prophet
from src.lstm_model import build_lstm_model, fit_lstm, predict_lstm
from src.evaluation import evaluate_model, compare_models
from src.visualisation import (
    plot_forecast_comparison,
    plot_metrics_comparison,
    plot_prophet_components,
    plot_training_history,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CSV = "data/sales_data.csv"
LOOK_BACK = 30  # days of history fed to LSTM
LSTM_EPOCHS = 50
LSTM_BATCH = 32
TEST_RATIO = 0.2
OUTPUT_DIR = "outputs"


def main() -> None:
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("=== 1. Generating / loading data ===")
    if not Path(DATA_CSV).exists():
        df = generate_sales_data()
        Path(DATA_CSV).parent.mkdir(exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
        print(f"  Generated {len(df)} rows → {DATA_CSV}")
    else:
        df = pd.read_csv(DATA_CSV, parse_dates=["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        print(f"  Loaded {len(df)} rows from {DATA_CSV}")

    train_df, test_df = train_test_split_ts(df, test_ratio=TEST_RATIO)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # ------------------------------------------------------------------
    # 2. Prophet
    # ------------------------------------------------------------------
    print("\n=== 2. Prophet model ===")
    prophet_model = fit_prophet(train_df)

    future = prophet_model.make_future_dataframe(periods=len(test_df))
    forecast = predict_prophet(prophet_model, future)

    # Align forecast to test dates
    prophet_forecast_test = forecast[forecast["ds"].isin(test_df["ds"])].copy()
    prophet_preds = prophet_forecast_test["yhat"].values
    actual_prophet = test_df["y"].values

    prophet_metrics = evaluate_model("Prophet", actual_prophet, prophet_preds)
    print(f"  {prophet_metrics}")

    plot_prophet_components(
        prophet_model, forecast,
        output_path=f"{OUTPUT_DIR}/prophet_components.png",
    )

    # ------------------------------------------------------------------
    # 3. LSTM
    # ------------------------------------------------------------------
    print("\n=== 3. LSTM model ===")
    train_vals = train_df["y"].values.astype(float)
    test_vals = test_df["y"].values.astype(float)

    scaled_train, scaled_test, scaler = scale_series(train_vals, test_vals)

    # Build sequences from the combined (train ++ test) scaled series so that
    # test sequences can look back into the training window.
    full_scaled = np.concatenate([scaled_train, scaled_test])
    X_train, y_train = create_sequences(scaled_train, LOOK_BACK)
    # For test sequences we need the LOOK_BACK tail of training + all test
    combined_for_test = np.concatenate([scaled_train[-LOOK_BACK:], scaled_test])
    X_test, y_test_scaled = create_sequences(combined_for_test, LOOK_BACK)

    lstm_model = build_lstm_model(look_back=LOOK_BACK)
    print(f"  Training LSTM (epochs={LSTM_EPOCHS}, batch={LSTM_BATCH}) …")
    history = fit_lstm(
        lstm_model, X_train, y_train,
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
        verbose=0,
    )

    lstm_preds = predict_lstm(lstm_model, X_test, scaler)
    actual_lstm = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    lstm_metrics = evaluate_model("LSTM", actual_lstm, lstm_preds)
    print(f"  {lstm_metrics}")

    plot_training_history(history, output_path=f"{OUTPUT_DIR}/lstm_training_history.png")

    # ------------------------------------------------------------------
    # 4. Comparison
    # ------------------------------------------------------------------
    print("\n=== 4. Model comparison ===")
    results = compare_models([prophet_metrics, lstm_metrics])
    print(results.to_string(index=False))
    results.to_csv(f"{OUTPUT_DIR}/metrics_comparison.csv", index=False)
    print(f"  Metrics saved → {OUTPUT_DIR}/metrics_comparison.csv")

    plot_metrics_comparison(results, output_path=f"{OUTPUT_DIR}/metrics_comparison.png")

    # ------------------------------------------------------------------
    # 5. Forecast comparison plot (aligned to common test window)
    # ------------------------------------------------------------------
    print("\n=== 5. Forecast comparison plot ===")
    # Use the shorter of the two prediction arrays for alignment
    n_common = min(len(prophet_preds), len(lstm_preds))
    common_dates = test_df["ds"].values[-n_common:]
    common_actual = actual_lstm[-n_common:]
    common_prophet = prophet_preds[-n_common:]
    common_lstm = lstm_preds[-n_common:]

    plot_forecast_comparison(
        pd.Series(common_dates),
        common_actual,
        common_prophet,
        common_lstm,
        output_path=f"{OUTPUT_DIR}/forecast_comparison.png",
    )

    print("\n✓ Pipeline complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

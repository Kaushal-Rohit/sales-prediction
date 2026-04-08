"""
main.py
-------
End-to-end pipeline: load data → Prophet → LSTM → compare results.
Run from the project root:
    python main.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for saving plots

warnings.filterwarnings('ignore')

# ── Make src/ importable ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import (
    load_and_clean, aggregate_monthly,
    train_test_split_ts, create_lstm_sequences, scale_series
)
from prophet_model import (
    train_prophet, forecast_prophet,
    evaluate_on_test, plot_prophet_forecast, plot_prophet_components
)
from lstm_model import (
    build_lstm_model, train_lstm, predict_lstm,
    plot_lstm_training, plot_lstm_forecast
)
from evaluate import evaluate_model, plot_comparison, plot_actual_vs_predicted

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "train.csv")
OUTPUT_DIR  = "outputs"
TEST_MONTHS = 6
LOOKBACK    = 12    # LSTM looks back 12 months

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\n" + "="*55)
    print("  Time Series Forecasting Pipeline")
    print("  Superstore Sales  |  Prophet vs LSTM")
    print("="*55 + "\n")

    # ── 1. Load & Preprocess ─────────────────────────────────────────────────
    print("[1/6] Loading and cleaning data...")
    df_raw    = load_and_clean(DATA_PATH)
    df_monthly = aggregate_monthly(df_raw)
    print(f"      Monthly records: {len(df_monthly)}")
    print(f"      Date range: {df_monthly['ds'].min().date()} → {df_monthly['ds'].max().date()}")

    train_df, test_df = train_test_split_ts(df_monthly, test_months=TEST_MONTHS)
    print(f"      Train: {len(train_df)} months | Test: {len(test_df)} months\n")

    # ── 2. Prophet ───────────────────────────────────────────────────────────
    print("[2/6] Training Prophet model...")
    prophet_model = train_prophet(train_df)
    forecast      = forecast_prophet(prophet_model, periods=TEST_MONTHS)

    plot_prophet_forecast(
        prophet_model, forecast, train_df, test_df,
        save_path=os.path.join(OUTPUT_DIR, "prophet_forecast.png")
    )
    plot_prophet_components(
        prophet_model, forecast,
        save_path=os.path.join(OUTPUT_DIR, "prophet_components.png")
    )

    prophet_preds  = evaluate_on_test(forecast, test_df)
    prophet_result = evaluate_model("Prophet", test_df['y'].values, prophet_preds)

    # ── 3. LSTM ──────────────────────────────────────────────────────────────
    print("[3/6] Preparing LSTM sequences...")
    train_vals = train_df['y'].values
    test_vals  = test_df['y'].values

    train_scaled, test_scaled, scaler = scale_series(train_vals, test_vals)

    # Build full scaled series for sequence extraction
    full_scaled = np.concatenate([train_scaled, test_scaled])
    split_idx   = len(train_scaled)

    X_all, y_all = create_lstm_sequences(full_scaled, lookback=LOOKBACK)

    # Sequences whose label falls in training range
    # The LOOKBACK-th element labels index LOOKBACK → split_idx-1 for train
    X_train = X_all[:split_idx - LOOKBACK]
    y_train = y_all[:split_idx - LOOKBACK]
    X_test  = X_all[split_idx - LOOKBACK:]
    y_test  = y_all[split_idx - LOOKBACK:]

    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}\n")

    print("[4/6] Building and training LSTM...")
    lstm_model = build_lstm_model(lookback=LOOKBACK)
    history    = train_lstm(lstm_model, X_train, y_train, epochs=150, batch_size=4)

    plot_lstm_training(
        history,
        save_path=os.path.join(OUTPUT_DIR, "lstm_training_history.png")
    )

    # ── 4. LSTM Predictions ──────────────────────────────────────────────────
    print("[5/6] Generating LSTM predictions...")
    lstm_preds  = predict_lstm(lstm_model, X_test, scaler)
    lstm_result = evaluate_model("LSTM", test_vals[-len(lstm_preds):], lstm_preds)

    plot_lstm_forecast(
        train_df['ds'], train_vals,
        test_df['ds'].iloc[-len(lstm_preds):], test_vals[-len(lstm_preds):],
        lstm_preds,
        save_path=os.path.join(OUTPUT_DIR, "lstm_forecast.png")
    )

    # ── 5. Compare ───────────────────────────────────────────────────────────
    print("[6/6] Comparing models...")
    common_len = min(len(prophet_preds), len(lstm_preds))
    common_dates  = test_df['ds'].values[-common_len:]
    common_actual = test_df['y'].values[-common_len:]

    plot_actual_vs_predicted(
        common_dates, common_actual,
        {"Prophet": prophet_preds[-common_len:], "LSTM": lstm_preds[-common_len:]},
        title="Sales Forecast — Prophet vs LSTM vs Actuals",
        save_path=os.path.join(OUTPUT_DIR, "model_comparison_overlay.png")
    )

    plot_comparison(
        [prophet_result, lstm_result],
        save_path=os.path.join(OUTPUT_DIR, "model_comparison_metrics.png")
    )

    # ── 6. Summary Table ─────────────────────────────────────────────────────
    summary = pd.DataFrame([prophet_result, lstm_result])
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*55)
    print("  FINAL RESULTS SUMMARY")
    print("="*55)
    print(summary.to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()

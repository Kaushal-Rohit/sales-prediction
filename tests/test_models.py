"""
Unit tests for the sales-prediction project.

Tests are intentionally lightweight and do NOT require a GPU or a live
internet connection.  Heavy TensorFlow / Prophet fits are replaced with
minimal configurations to keep the test suite fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure the repo root is importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# data.generate_data
# ---------------------------------------------------------------------------
from data.generate_data import generate_sales_data


class TestGenerateData:
    def test_returns_dataframe(self):
        df = generate_sales_data(start="2022-01-01", end="2022-03-31")
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = generate_sales_data(start="2022-01-01", end="2022-03-31")
        assert set(df.columns) == {"ds", "y"}

    def test_length(self):
        df = generate_sales_data(start="2022-01-01", end="2022-12-31")
        assert len(df) == 365

    def test_no_negative_values(self):
        df = generate_sales_data(start="2020-01-01", end="2023-12-31")
        assert (df["y"] >= 0).all()

    def test_sorted_dates(self):
        df = generate_sales_data(start="2022-01-01", end="2022-06-30")
        assert df["ds"].is_monotonic_increasing

    def test_reproducibility(self):
        df1 = generate_sales_data(seed=0)
        df2 = generate_sales_data(seed=0)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# src.data_preprocessing
# ---------------------------------------------------------------------------
from src.data_preprocessing import (
    load_data,
    train_test_split_ts,
    create_sequences,
    scale_series,
)


class TestPreprocessing:
    @pytest.fixture()
    def sample_df(self, tmp_path):
        df = generate_sales_data(start="2020-01-01", end="2021-12-31", seed=1)
        csv = tmp_path / "test_sales.csv"
        df.to_csv(csv, index=False)
        return csv, df

    def test_load_data(self, sample_df):
        csv, original = sample_df
        loaded = load_data(str(csv))
        assert isinstance(loaded["ds"].iloc[0], pd.Timestamp)
        assert len(loaded) == len(original)

    def test_train_test_split_ratio(self, sample_df):
        _, df = sample_df
        train, test = train_test_split_ts(df, test_ratio=0.2)
        assert len(train) + len(test) == len(df)
        assert len(test) == pytest.approx(len(df) * 0.2, abs=1)

    def test_train_before_test(self, sample_df):
        _, df = sample_df
        train, test = train_test_split_ts(df, test_ratio=0.2)
        assert train["ds"].max() < test["ds"].min()

    def test_create_sequences_shapes(self):
        series = np.arange(100, dtype=float)
        X, y = create_sequences(series, look_back=10)
        assert X.shape == (90, 10, 1)
        assert y.shape == (90,)

    def test_create_sequences_values(self):
        series = np.arange(20, dtype=float)
        X, y = create_sequences(series, look_back=5)
        np.testing.assert_array_equal(X[0, :, 0], np.arange(5))
        assert y[0] == 5.0

    def test_scale_series_range(self):
        train = np.linspace(100, 500, 200)
        test = np.linspace(450, 600, 50)
        s_train, s_test, scaler = scale_series(train, test)
        assert s_train.min() >= 0.0
        assert s_train.max() <= 1.0

    def test_scale_series_inverse(self):
        train = np.linspace(100, 500, 200)
        test = np.linspace(450, 600, 50)
        _, _, scaler = scale_series(train, test)
        original_sample = np.array([[300.0]])
        scaled = scaler.transform(original_sample)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, original_sample, atol=1e-6)


# ---------------------------------------------------------------------------
# src.evaluation
# ---------------------------------------------------------------------------
from src.evaluation import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    evaluate_model,
    compare_models,
)


class TestEvaluation:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(
            (2 + 2 + 3) / 3
        )

    def test_mse_perfect(self):
        y = np.ones(5)
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            np.sqrt((9 + 16) / 2)
        )

    def test_mape_perfect(self):
        y = np.array([100.0, 200.0])
        assert mean_absolute_percentage_error(y, y) == pytest.approx(0.0)

    def test_mape_ignores_zero_actuals(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([999.0, 110.0])
        # Only the second element contributes
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)

    def test_evaluate_model_keys(self):
        y = np.array([100.0, 200.0, 300.0])
        result = evaluate_model("TestModel", y, y)
        assert result["Model"] == "TestModel"
        for key in ("MAE", "RMSE", "MAPE (%)"):
            assert key in result

    def test_compare_models_sorted_by_rmse(self):
        r1 = evaluate_model("A", np.array([100.0]), np.array([120.0]))
        r2 = evaluate_model("B", np.array([100.0]), np.array([105.0]))
        df = compare_models([r1, r2])
        assert df.iloc[0]["Model"] == "B"

    def test_compare_models_dataframe(self):
        r1 = evaluate_model("A", np.ones(5) * 100, np.ones(5) * 110)
        df = compare_models([r1])
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Model", "MAE", "RMSE", "MAPE (%)"]


# ---------------------------------------------------------------------------
# src.prophet_model  (lightweight – avoids a full fit on large data)
# ---------------------------------------------------------------------------
from src.prophet_model import build_prophet_model, fit_prophet, predict_prophet


class TestProphetModel:
    # Small dataset constants for lightweight Prophet testing
    _TRAIN_PERIODS = 120
    _BASE_SALES = 500
    _TREND_RANGE = 100
    _NOISE_STD = 10
    _FUTURE_PERIODS = 10

    @pytest.fixture()
    def small_train_df(self):
        dates = pd.date_range("2021-01-01", periods=self._TRAIN_PERIODS, freq="D")
        y = (
            self._BASE_SALES
            + np.linspace(0, self._TREND_RANGE, self._TRAIN_PERIODS)
            + np.random.default_rng(7).normal(0, self._NOISE_STD, self._TRAIN_PERIODS)
        )
        return pd.DataFrame({"ds": dates, "y": y})

    def test_build_returns_prophet(self):
        from prophet import Prophet

        model = build_prophet_model()
        assert isinstance(model, Prophet)

    def test_fit_predict_shape(self, small_train_df):
        model = fit_prophet(small_train_df)
        future = model.make_future_dataframe(periods=self._FUTURE_PERIODS)
        forecast = predict_prophet(model, future)
        assert "yhat" in forecast.columns
        assert len(forecast) == len(small_train_df) + self._FUTURE_PERIODS


# ---------------------------------------------------------------------------
# src.lstm_model  (minimal architecture test – no actual training)
# ---------------------------------------------------------------------------
from src.lstm_model import build_lstm_model, predict_lstm


class TestLSTMModel:
    def test_model_output_shape(self):
        model = build_lstm_model(look_back=10, units=8)
        X = np.random.rand(20, 10, 1).astype("float32")
        out = model.predict(X, verbose=0)
        assert out.shape == (20, 1)

    def test_predict_lstm_inverse_transform(self):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        dummy = np.linspace(100, 500, 200).reshape(-1, 1)
        scaler.fit(dummy)

        model = build_lstm_model(look_back=5, units=4)
        X = np.random.rand(10, 5, 1).astype("float32")
        preds = predict_lstm(model, X, scaler)
        assert preds.shape == (10,)
        # All predictions should be within a reasonable range
        assert preds.min() >= 50
        assert preds.max() <= 600

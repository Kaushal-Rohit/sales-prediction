"""
tests/test_rf_model.py
----------------------
Unit tests for the Random Forest sales-prediction module (src/rf_model.py)
and the end-to-end train_weighted_rf pipeline.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rf_model import (
    preprocess_data,
    compute_sample_weights,
    train_baseline_rf,
    train_weighted_rf,
    evaluate_model,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    WEIGHTED_FEATURES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_csv(tmp_path: "Path", n: int = 200) -> str:
    """Create a small synthetic CSV with the same schema as train.csv."""
    rng = np.random.default_rng(0)
    regions = ["South", "West", "Central", "East"]
    ship_modes = ["Standard Class", "First Class", "Second Class", "Same Day"]
    segments = ["Consumer", "Corporate", "Home Office"]
    categories = ["Furniture", "Office Supplies", "Technology"]
    sub_categories = ["Chairs", "Paper", "Phones", "Binders", "Bookcases"]

    dates = pd.date_range("2017-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "Row ID": range(1, n + 1),
            "Order ID": [f"CA-{i}" for i in range(n)],
            "Order Date": dates.strftime("%d/%m/%Y"),
            "Ship Date": dates.strftime("%d/%m/%Y"),
            "Ship Mode": rng.choice(ship_modes, n),
            "Customer ID": [f"CG-{i}" for i in range(n)],
            "Customer Name": [f"Name {i}" for i in range(n)],
            "Segment": rng.choice(segments, n),
            "Country": "United States",
            "City": "Chicago",
            "State": "Illinois",
            "Postal Code": "60601",
            "Region": rng.choice(regions, n),
            "Product ID": [f"FUR-{i}" for i in range(n)],
            "Category": rng.choice(categories, n),
            "Sub-Category": rng.choice(sub_categories, n),
            "Product Name": [f"Product {i}" for i in range(n)],
            "Sales": rng.exponential(scale=200, size=n).clip(1),
        }
    )

    csv_path = str(tmp_path / "train_test.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def csv_path(tmp_path):
    return _make_csv(tmp_path)


@pytest.fixture()
def xy(csv_path):
    X, y, encoders = preprocess_data(csv_path)
    return X, y, encoders


# ── preprocess_data ────────────────────────────────────────────────────────────

class TestPreprocessData:
    def test_returns_tuple(self, csv_path):
        result = preprocess_data(csv_path)
        assert isinstance(result, tuple) and len(result) == 3

    def test_X_is_dataframe(self, xy):
        X, _, _ = xy
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, xy):
        _, y, _ = xy
        assert isinstance(y, pd.Series)

    def test_no_sales_in_X(self, xy):
        X, _, _ = xy
        assert "Sales" not in X.columns

    def test_temporal_features_present(self, xy):
        X, _, _ = xy
        for col in ("Order_Month", "Order_Quarter", "Order_Year", "Order_DayOfWeek"):
            assert col in X.columns, f"{col} missing from X"

    def test_order_date_dropped(self, xy):
        X, _, _ = xy
        assert "Order Date" not in X.columns

    def test_all_numeric(self, xy):
        X, _, _ = xy
        assert X.select_dtypes(include="number").shape[1] == X.shape[1]

    def test_no_nulls(self, xy):
        X, y, _ = xy
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_encoders_for_all_categoricals(self, xy):
        _, _, encoders = xy
        for col in CATEGORICAL_COLS:
            assert col in encoders

    def test_positive_sales(self, xy):
        _, y, _ = xy
        assert (y > 0).all()


# ── compute_sample_weights ────────────────────────────────────────────────────

class TestComputeSampleWeights:
    def test_length_matches_training_set(self, xy):
        X, y, _ = xy
        weights = compute_sample_weights(X, y)
        assert len(weights) == len(X)

    def test_weights_positive(self, xy):
        X, y, _ = xy
        weights = compute_sample_weights(X, y)
        assert (weights > 0).all()

    def test_weights_mean_approx_one(self, xy):
        X, y, _ = xy
        weights = compute_sample_weights(X, y)
        assert abs(weights.mean() - 1.0) < 1e-6

    def test_higher_factor_increases_spread(self, xy):
        X, y, _ = xy
        w_low  = compute_sample_weights(X, y, weight_factor=1.0)
        w_high = compute_sample_weights(X, y, weight_factor=5.0)
        assert w_high.std() >= w_low.std()

    def test_returns_ndarray(self, xy):
        X, y, _ = xy
        weights = compute_sample_weights(X, y)
        assert isinstance(weights, np.ndarray)


# ── train_baseline_rf ──────────────────────────────────────────────────────────

class TestTrainBaselineRF:
    def test_returns_fitted_model(self, xy):
        from sklearn.ensemble import RandomForestRegressor
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        assert isinstance(model, RandomForestRegressor)

    def test_predict_shape(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_feature_importances_sum_to_one(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6


# ── train_weighted_rf ──────────────────────────────────────────────────────────

class TestTrainWeightedRF:
    def test_returns_model_and_weights(self, xy):
        X, y, _ = xy
        model, weights = train_weighted_rf(X, y)
        assert model is not None
        assert weights is not None

    def test_weights_length(self, xy):
        X, y, _ = xy
        _, weights = train_weighted_rf(X, y)
        assert len(weights) == len(X)

    def test_predict_shape(self, xy):
        X, y, _ = xy
        model, _ = train_weighted_rf(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_feature_importances_sum_to_one(self, xy):
        X, y, _ = xy
        model, _ = train_weighted_rf(X, y)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6


# ── evaluate_model ────────────────────────────────────────────────────────────

class TestEvaluateModel:
    def test_returns_dict_with_required_keys(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        result = evaluate_model("Test", model, X, y)
        for key in ("model", "MAE", "MSE", "R2", "predictions"):
            assert key in result

    def test_model_name_stored(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        result = evaluate_model("MyModel", model, X, y)
        assert result["model"] == "MyModel"

    def test_mae_non_negative(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        result = evaluate_model("Test", model, X, y)
        assert result["MAE"] >= 0

    def test_mse_non_negative(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        result = evaluate_model("Test", model, X, y)
        assert result["MSE"] >= 0

    def test_predictions_length(self, xy):
        X, y, _ = xy
        model = train_baseline_rf(X, y)
        result = evaluate_model("Test", model, X, y)
        assert len(result["predictions"]) == len(X)

    def test_perfect_predictions_zero_error(self, xy):
        """When predictions equal actuals, MAE and MSE should be 0."""
        from sklearn.ensemble import RandomForestRegressor
        import unittest.mock as mock

        X, y, _ = xy
        model = train_baseline_rf(X, y)
        # Mock predict to return exact values
        with mock.patch.object(model, "predict", return_value=y.values):
            result = evaluate_model("Test", model, X, y)
        assert result["MAE"] == pytest.approx(0.0, abs=1e-6)
        assert result["MSE"] == pytest.approx(0.0, abs=1e-6)
        assert result["R2"]  == pytest.approx(1.0, abs=1e-6)

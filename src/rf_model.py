"""
rf_model.py
-----------
Random Forest training utilities for sales prediction.

Provides:
- preprocess_data        : Load and encode features from the raw CSV.
- compute_sample_weights : Build per-sample weights that emphasise Region
                           and Ship Mode by up-weighting high-variance
                           feature groups.
- train_baseline_rf      : Fit a RandomForestRegressor with equal weights.
- train_weighted_rf      : Fit a RandomForestRegressor with Region / Ship Mode
                           emphasis weights.
- evaluate_model         : Return MAE, MSE, and R² for a fitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Order Date",
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
]

CATEGORICAL_COLS = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]

WEIGHTED_FEATURES = ["Region", "Ship Mode"]

RF_PARAMS = dict(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)


# ── Data loading & preprocessing ──────────────────────────────────────────────

def preprocess_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Load *csv_path*, select relevant columns, engineer temporal features,
    label-encode categoricals, and return ``(X, y, encoders)``.

    Parameters
    ----------
    csv_path : str
        Path to the raw ``train.csv`` file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric, ready for sklearn).
    y : pd.Series
        Sales target.
    encoders : dict[str, LabelEncoder]
        Mapping of column name → fitted LabelEncoder for every categorical.
    """
    df = pd.read_csv(csv_path)
    df = df[FEATURE_COLS].copy()

    # ── Temporal features from Order Date ────────────────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Quarter"] = df["Order Date"].dt.quarter
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek
    df.drop(columns=["Order Date"], inplace=True)

    # ── Label-encode categoricals ─────────────────────────────────────────────
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df.dropna(inplace=True)

    y = df.pop("Sales")
    X = df

    return X, y, encoders


# ── Sample-weight computation ─────────────────────────────────────────────────

def compute_sample_weights(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weight_factor: float = 3.0,
    weighted_features: list[str] | None = None,
) -> np.ndarray:
    """Return per-sample weights that emphasise *weighted_features*.

    Samples whose ``(Region, Ship Mode)`` group exhibits higher sales
    variance receive a proportionally larger weight, directing the model to
    fit those groups more accurately.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (post-encoding).
    y_train : pd.Series
        Training target (Sales).
    weight_factor : float
        Multiplier applied to the normalised group variance.
        A value of 3.0 means high-variance groups are weighted
        up to 4× more than low-variance groups (1 + 3×1 = 4).
    weighted_features : list[str] | None
        Column names to group by.  Defaults to ``["Region", "Ship Mode"]``.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Non-negative sample weights with mean ≈ 1.
    """
    if weighted_features is None:
        weighted_features = WEIGHTED_FEATURES

    df_tmp = X_train.copy()
    df_tmp["_y"] = y_train.values

    group_var = (
        df_tmp.groupby(weighted_features)["_y"]
        .var()
        .reset_index()
        .rename(columns={"_y": "_group_var"})
    )
    df_tmp = df_tmp.merge(group_var, on=weighted_features, how="left")
    df_tmp["_group_var"] = df_tmp["_group_var"].fillna(0.0)

    var_min = df_tmp["_group_var"].min()
    var_max = df_tmp["_group_var"].max()

    if var_max > var_min:
        norm_var = (df_tmp["_group_var"] - var_min) / (var_max - var_min)
    else:
        norm_var = np.zeros(len(df_tmp))

    weights = 1.0 + weight_factor * norm_var.values
    # Normalise so mean == 1 (keeps effective sample size stable)
    weights = weights / weights.mean()
    return weights


# ── Model training ─────────────────────────────────────────────────────────────

def train_baseline_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    """Train a Random Forest with equal sample weights (baseline)."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_weighted_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weight_factor: float = 3.0,
) -> tuple[RandomForestRegressor, np.ndarray]:
    """Train a Random Forest that emphasises Region and Ship Mode.

    Parameters
    ----------
    X_train, y_train : training data.
    weight_factor : float
        Controls how much Region / Ship Mode are emphasised.

    Returns
    -------
    model : fitted RandomForestRegressor
    sample_weights : np.ndarray of weights used during training
    """
    sample_weights = compute_sample_weights(X_train, y_train, weight_factor=weight_factor)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model, sample_weights


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Return a metrics dict for *model* on the hold-out set.

    Returns
    -------
    dict with keys: ``model``, ``MAE``, ``MSE``, ``R2``, ``predictions``.
    """
    preds = model.predict(X_test)
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_test, preds)),
        "MSE": float(mean_squared_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
        "predictions": preds,
    }

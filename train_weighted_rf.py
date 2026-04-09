"""
train_weighted_rf.py
--------------------
Train and compare two Random Forest models for sales prediction:

  * Baseline model  – equal sample weights (standard RF).
  * Weighted model  – sample weights emphasise Region and Ship Mode feature
                      groups, so the model pays more attention to high-variance
                      Region / Ship Mode combinations.

Run from the project root:
    python train_weighted_rf.py

Outputs (saved to outputs/):
  - feature_importance_comparison.png
  - prediction_comparison.png
  - performance_metrics_comparison.png
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rf_model import (
    preprocess_data,
    train_baseline_rf,
    train_weighted_rf,
    evaluate_model,
    WEIGHTED_FEATURES,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join("data", "train.csv")
OUTPUT_DIR     = "outputs"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
WEIGHT_FACTOR  = 3.0          # weight multiplier for Region / Ship Mode groups

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _savefig(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"      Saved: {path}")


def plot_feature_importance(
    baseline_model,
    weighted_model,
    feature_names: list[str],
) -> None:
    """Side-by-side horizontal bar chart of feature importances."""
    baseline_imp = baseline_model.feature_importances_
    weighted_imp = weighted_model.feature_importances_

    # Sort by mean importance across both models
    order = np.argsort((baseline_imp + weighted_imp) / 2)
    feat_sorted   = [feature_names[i] for i in order]
    base_sorted   = baseline_imp[order]
    weight_sorted = weighted_imp[order]

    x = np.arange(len(feat_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.barh(x - width / 2, base_sorted,   width, label="Baseline", color="#4C72B0")
    bars2 = ax.barh(x + width / 2, weight_sorted, width, label="Weighted (Region & Ship Mode)", color="#DD8452")

    ax.set_yticks(x)
    ax.set_yticklabels(feat_sorted, fontsize=10)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title("Feature Importance: Baseline vs Weighted Model")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Annotate the two weighted features
    for i, feat in enumerate(feat_sorted):
        if feat in WEIGHTED_FEATURES:
            ax.annotate(
                "↑ emphasised",
                xy=(weight_sorted[i], i + width / 2),
                xytext=(weight_sorted[i] + 0.005, i + width / 2),
                fontsize=8,
                color="#DD8452",
            )

    _savefig("feature_importance_comparison.png")


def plot_predictions(
    y_test: np.ndarray,
    baseline_preds: np.ndarray,
    weighted_preds: np.ndarray,
) -> None:
    """Scatter plot: Actual vs Predicted for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    max_val = max(y_test.max(), baseline_preds.max(), weighted_preds.max()) * 1.05
    lim = (0, max_val)

    for ax, preds, label, color in zip(
        axes,
        [baseline_preds, weighted_preds],
        ["Baseline Model", "Weighted Model (Region & Ship Mode)"],
        ["#4C72B0", "#DD8452"],
    ):
        ax.scatter(y_test, preds, alpha=0.3, s=10, color=color)
        ax.plot(lim, lim, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Actual Sales ($)")
        ax.set_ylabel("Predicted Sales ($)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Actual vs Predicted Sales", fontsize=13, fontweight="bold")
    _savefig("prediction_comparison.png")


def plot_metrics(baseline_result: dict, weighted_result: dict) -> None:
    """Grouped bar chart comparing MAE, MSE (√), and R² for both models."""
    metrics = ["MAE", "√MSE", "R²"]

    base_vals   = [
        baseline_result["MAE"],
        np.sqrt(baseline_result["MSE"]),
        baseline_result["R2"],
    ]
    weight_vals = [
        weighted_result["MAE"],
        np.sqrt(weighted_result["MSE"]),
        weighted_result["R2"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, base_vals,   width, label="Baseline",  color="#4C72B0")
    bars2 = ax.bar(x + width / 2, weight_vals, width, label="Weighted",  color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Metric Value")
    ax.set_title("Performance Metrics: Baseline vs Weighted Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)

    _savefig("performance_metrics_comparison.png")


# ── Analysis ──────────────────────────────────────────────────────────────────

def print_analysis(baseline_result: dict, weighted_result: dict) -> None:
    """Print a human-readable comparison summary."""
    b = baseline_result
    w = weighted_result

    mae_diff  = b["MAE"] - w["MAE"]
    mse_diff  = b["MSE"] - w["MSE"]
    r2_diff   = w["R2"]  - b["R2"]

    mae_pct   = (mae_diff / b["MAE"]) * 100 if b["MAE"] else 0.0
    rmse_pct  = ((np.sqrt(b["MSE"]) - np.sqrt(w["MSE"])) / np.sqrt(b["MSE"])) * 100 if b["MSE"] else 0.0

    improved = mae_diff > 0 and mse_diff > 0

    bar = "=" * 60

    print(f"\n{bar}")
    print("  DETAILED ANALYSIS: Region & Ship Mode Weighting Impact")
    print(bar)
    print(f"\n{'Metric':<18} {'Baseline':>12} {'Weighted':>12} {'Δ (B→W)':>14}")
    print("-" * 58)
    print(f"{'MAE':<18} {b['MAE']:>12.2f} {w['MAE']:>12.2f} {mae_diff:>+13.2f}")
    print(f"{'RMSE':<18} {np.sqrt(b['MSE']):>12.2f} {np.sqrt(w['MSE']):>12.2f} {np.sqrt(b['MSE'])-np.sqrt(w['MSE']):>+13.2f}")
    print(f"{'MSE':<18} {b['MSE']:>12.2f} {w['MSE']:>12.2f} {mse_diff:>+13.2f}")
    print(f"{'R² Score':<18} {b['R2']:>12.4f} {w['R2']:>12.4f} {r2_diff:>+13.4f}")
    print("-" * 58)

    threshold = 1.0     # % improvement to consider "significant"

    print(f"\n📊 Summary:")
    print(f"   MAE improvement  : {mae_pct:+.2f}%")
    print(f"   RMSE improvement : {rmse_pct:+.2f}%")
    print(f"   R² improvement   : {r2_diff:+.4f}")

    print(f"\n🔍 Conclusion:")
    if improved and (abs(mae_pct) >= threshold or abs(rmse_pct) >= threshold):
        print(
            f"   ✅ Giving high weight to Region and Ship Mode "
            f"DOES make a significant difference.\n"
            f"   MAE improved by {mae_pct:.2f}% and RMSE by {rmse_pct:.2f}%.\n"
            f"   The weighted model is BETTER at predicting sales."
        )
    elif improved:
        print(
            f"   ⚠️  Giving high weight to Region and Ship Mode gives a marginal\n"
            f"   improvement (MAE: {mae_pct:.2f}%, RMSE: {rmse_pct:.2f}%), but the\n"
            f"   difference is below the {threshold}% significance threshold."
        )
    else:
        print(
            f"   ❌ Giving high weight to Region and Ship Mode does NOT improve\n"
            f"   predictions in this case.\n"
            f"   The baseline model performs equally well or better."
        )

    print(f"\n💡 Interpretation:")
    print(
        "   Sample-weighting emphasises high-variance Region/Ship Mode groups,\n"
        "   pushing the Random Forest to fit those combinations more accurately.\n"
        "   Whether this helps depends on how strongly Region and Ship Mode\n"
        "   drive sales variation in the underlying data."
    )
    print(f"\n{bar}\n")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Sales Prediction — Baseline vs Weighted Random Forest")
    print("  Evaluating impact of high weights on Region & Ship Mode")
    print("=" * 60 + "\n")

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    print("[1/5] Loading and preprocessing data...")
    X, y, encoders = preprocess_data(DATA_PATH)
    print(f"      Rows: {len(X):,}  |  Features: {X.shape[1]}")
    print(f"      Feature columns: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # ── 2. Train baseline model ───────────────────────────────────────────────
    print("[2/5] Training Baseline Random Forest (equal weights)...")
    baseline_model = train_baseline_rf(X_train, y_train)
    print("      Done.\n")

    # ── 3. Train weighted model ───────────────────────────────────────────────
    print(f"[3/5] Training Weighted Random Forest "
          f"(weight_factor={WEIGHT_FACTOR} for {WEIGHTED_FEATURES})...")
    weighted_model, sample_weights = train_weighted_rf(
        X_train, y_train, weight_factor=WEIGHT_FACTOR
    )
    print(f"      Sample weights — min: {sample_weights.min():.3f}, "
          f"max: {sample_weights.max():.3f}, mean: {sample_weights.mean():.3f}\n")

    # ── 4. Evaluate both models ───────────────────────────────────────────────
    print("[4/5] Evaluating models on hold-out test set...")
    baseline_result = evaluate_model("Baseline", baseline_model, X_test, y_test)
    weighted_result = evaluate_model("Weighted", weighted_model, X_test, y_test)
    print(f"      Baseline → MAE: {baseline_result['MAE']:.2f}, "
          f"RMSE: {np.sqrt(baseline_result['MSE']):.2f}, "
          f"R²: {baseline_result['R2']:.4f}")
    print(f"      Weighted → MAE: {weighted_result['MAE']:.2f}, "
          f"RMSE: {np.sqrt(weighted_result['MSE']):.2f}, "
          f"R²: {weighted_result['R2']:.4f}\n")

    # ── 5. Visualise ─────────────────────────────────────────────────────────
    print("[5/5] Generating visualisations...")
    feature_names = list(X.columns)
    plot_feature_importance(baseline_model, weighted_model, feature_names)
    plot_predictions(
        y_test.values,
        baseline_result["predictions"],
        weighted_result["predictions"],
    )
    plot_metrics(baseline_result, weighted_result)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print_analysis(baseline_result, weighted_result)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary = pd.DataFrame(
        [
            {
                "Model": baseline_result["model"],
                "MAE":   baseline_result["MAE"],
                "MSE":   baseline_result["MSE"],
                "RMSE":  np.sqrt(baseline_result["MSE"]),
                "R2":    baseline_result["R2"],
            },
            {
                "Model": weighted_result["model"],
                "MAE":   weighted_result["MAE"],
                "MSE":   weighted_result["MSE"],
                "RMSE":  np.sqrt(weighted_result["MSE"]),
                "R2":    weighted_result["R2"],
            },
        ]
    )
    csv_path = os.path.join(OUTPUT_DIR, "rf_results_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Results summary saved to: {csv_path}")
    print(f"All visualisations saved to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()

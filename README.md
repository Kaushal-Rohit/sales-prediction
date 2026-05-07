# Sales Prediction – Multi-Model Time-Series Forecasting

A practical machine-learning project that uses historical sales data to predict
future trends.  Two complementary forecasting approaches are implemented and
compared:

| Approach | Library | Strengths | Weaknesses |
|---|---|---|---|
| **Prophet** | `prophet` (Meta) | Interpretable components, handles seasonality out of the box, robust to missing data | May under-fit complex non-linear patterns |
| **LSTM** | `tensorflow` / `keras` | Captures long-range non-linear dependencies | Less interpretable, needs more data and tuning |

---

## Project Structure

```
sales-prediction/
├── data/
│   ├── generate_data.py      # Synthetic retail sales data generator
│   └── sales_data.csv        # Generated dataset (created on first run)
├── src/
│   ├── data_preprocessing.py # Load, split, scale, sequence utilities
│   ├── prophet_model.py      # Prophet wrapper
│   ├── lstm_model.py         # LSTM model (TensorFlow/Keras)
│   ├── evaluation.py         # MAE / RMSE / MAPE metrics & comparison
│   └── visualisation.py      # Plotting helpers
├── tests/
│   └── test_models.py        # Pytest unit tests (26 tests)
├── outputs/                  # Generated plots & CSV (created on first run)
│   ├── forecast_comparison.png
│   ├── metrics_comparison.png
│   ├── prophet_components.png
│   ├── lstm_training_history.png
│   └── metrics_comparison.csv
├── main.py                   # End-to-end pipeline
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
1. Generate 6 years of synthetic daily retail sales data (2018–2023).
2. Split 80 % / 20 % into train / test sets.
3. Fit a **Prophet** model and evaluate it on the test set.
4. Build and train a stacked **LSTM** model and evaluate it.
5. Print a comparison table (MAE, RMSE, MAPE %).
6. Write four plots + a metrics CSV to `outputs/`.

### 3. Run the tests

```bash
pytest tests/ -v
```

---

## Dataset

The synthetic dataset (`data/generate_data.py`) mimics a retail store:

* **Long-term trend** – daily sales grow from ~500 to ~1 500 over 6 years.
* **Weekly seasonality** – higher sales on Fridays and Saturdays.
* **Yearly seasonality** – November–December holiday bump.
* **Monthly oscillation** – subtle 30-day cycle.
* **Gaussian noise** – standard deviation ≈ 80 units.

---

## Model Trade-offs

### Prophet

* **Interpretable** – the model explicitly decomposes predictions into trend,
  weekly, and yearly seasonality components (see `outputs/prophet_components.png`).
* **Fast to train** – fits in seconds even on multi-year daily data.
* **Handles missing data** gracefully and does not require scaling.
* **Limitation** – assumes additive (or multiplicative) structure; may miss
  complex sequential dependencies.

### LSTM

* **Flexible** – learns arbitrary non-linear temporal patterns from raw
  sequences.
* **Scalable** – benefits from more data and GPU acceleration.
* **Limitation** – acts as a black box; requires careful hyper-parameter
  tuning, scaling, and a `look_back` window design.

### Sample Results (test set)

| Model   | MAE     | RMSE     | MAPE (%) |
|---------|---------|----------|----------|
| Prophet | ~87     | ~108     | ~5.7     |
| LSTM    | ~131    | ~166     | ~8.3     |

*Exact numbers will vary due to random initialization in LSTM training.*

For this structured dataset with clear seasonality, Prophet outperforms LSTM
because the additive model assumptions align well with how the data was
generated.  On noisier or more complex datasets, LSTM may close the gap.

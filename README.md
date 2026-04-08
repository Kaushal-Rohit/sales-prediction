# 📈 Time Series Forecasting: Sales Prediction using Prophet & LSTM

> A practical machine learning project that forecasts future sales trends using historical Superstore data — comparing **Facebook Prophet** and **LSTM (Long Short-Term Memory)** models across accuracy and interpretability dimensions.

---

## 🗂️ Project Structure

```
time-series-forecasting/
│
├── data/
│   └── train.csv                  # Raw Superstore sales dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Prophet_Model.ipynb     # Prophet forecasting
│   ├── 03_LSTM_Model.ipynb        # LSTM forecasting
│   └── 04_Model_Comparison.ipynb  # Side-by-side comparison & trade-offs
│
├── src/
│   ├── data_preprocessing.py      # Data loading & cleaning utilities
│   ├── prophet_model.py           # Prophet training & forecasting
│   ├── lstm_model.py              # LSTM training & forecasting
│   └── evaluate.py                # Shared evaluation metrics (MAE, RMSE, MAPE)
│
├── outputs/                       # Saved plots and model results
├── main.py                        # Run full pipeline from CLI
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** Superstore Sales Dataset (Kaggle)
- **Rows:** 9,800 orders
- **Date Range:** 2015–2018
- **Target Variable:** `Sales` (aggregated monthly)
- **Key Features Used:** `Order Date`, `Sales`, `Category`, `Region`

---

## 🧠 Models Compared

| Feature              | Prophet                         | LSTM                            |
|----------------------|----------------------------------|----------------------------------|
| Type                 | Additive decomposition model     | Recurrent Neural Network         |
| Interpretability     | ⭐⭐⭐⭐⭐ (high — explainable)   | ⭐⭐ (low — black box)            |
| Training Speed       | Fast (seconds)                   | Slow (requires GPU for best)     |
| Handles Seasonality  | Built-in (daily/weekly/yearly)   | Learned from data                |
| Data Required        | Works on small datasets          | Needs large sequences            |
| Best For             | Business forecasting, trends     | Complex non-linear patterns      |
| Accuracy (this data) | MAE: ~8,200 | MAE: ~7,400 (after tuning)    |

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/time-series-forecasting.git
cd time-series-forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Place `train.csv` inside the `data/` folder.

### 4. Run Full Pipeline (CLI)
```bash
python main.py
```

### 5. Or Explore Notebooks
```bash
jupyter notebook notebooks/
```

---

## 📉 Results & Visualizations

All output plots are saved to the `outputs/` directory after running the pipeline:

- `monthly_sales_trend.png` — historical sales decomposition
- `prophet_forecast.png` — Prophet 90-day forecast with confidence intervals
- `lstm_forecast.png` — LSTM predictions vs actuals
- `model_comparison.png` — side-by-side MAE / RMSE bar chart

---

## 🔍 Key Findings

1. **Prophet** excels at capturing yearly seasonality and holiday effects with zero tuning — ideal for business stakeholders who need explainable forecasts.
2. **LSTM** achieves slightly better raw accuracy on this dataset after hyperparameter tuning, but at the cost of interpretability and training complexity.
3. Monthly aggregation significantly smooths noise and improves both models' performance compared to daily-level data.

---

## 🛠️ Tech Stack

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- prophet (Facebook/Meta)
- TensorFlow / Keras (LSTM)

---

## 📚 References

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [Keras LSTM Time Series Tutorial](https://keras.io/examples/timeseries/)
- [Superstore Dataset — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

---

## 👤 Author

**Kaushal** — CSE Student at KPGU  
Internship Project | Python & ML Enthusiast

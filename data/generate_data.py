"""
Generates synthetic retail sales data with realistic trend and seasonality components.

The dataset mimics a retail store with:
  - A long-term upward trend.
  - Weekly seasonality (higher sales on Fridays and Saturdays).
  - Yearly seasonality (holiday season peak in November/December).
  - Gaussian noise.
"""

import numpy as np
import pandas as pd


MIN_DAILY_SALES = 50  # minimum realistic daily sales floor (in revenue units)


def generate_sales_data(
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``ds`` (date) and ``y`` (daily sales)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    # Long-term linear trend
    trend = np.linspace(500, 1500, n)

    # Weekly seasonality: higher on Friday (4) and Saturday (5)
    day_of_week = dates.dayofweek.to_numpy()
    weekly = np.where(day_of_week == 5, 300, np.where(day_of_week == 4, 200, 0))

    # Yearly seasonality: holiday bump in Nov-Dec (months 11 and 12)
    day_of_year = dates.dayofyear.to_numpy()
    yearly = 400 * np.sin(2 * np.pi * (day_of_year - 275) / 365) * (
        (dates.month >= 11).astype(float)
    )

    # Small monthly oscillation
    monthly = 100 * np.sin(2 * np.pi * day_of_year / 30.5)

    # Gaussian noise
    noise = rng.normal(0, 80, n)

    sales = trend + weekly + yearly + monthly + noise
    # Clip to realistic non-negative values
    sales = np.clip(sales, MIN_DAILY_SALES, None)

    df = pd.DataFrame({"ds": dates, "y": np.round(sales, 2)})
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    out_path = "data/sales_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())

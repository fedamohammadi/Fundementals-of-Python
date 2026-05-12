"""
Data Cleaning — Missing Values and Duplicates:
- Detecting missing values: isnull, notna, per-column counts
- Dropping missing values: dropna with axis, how, thresh, subset
- Filling missing values: fillna, ffill, bfill, interpolate
- Detecting and removing duplicates: duplicated, drop_duplicates
- Fixing inconsistent data: string normalisation, categorical standardisation
- Detecting and handling outliers: IQR method and Z-score
- Practical example: cleaning a messy employee survey dataset
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: intentionally messy survey records
# ==============================================================
# This dataset is used across all sections so each demo builds
# on the same realistic context rather than inventing new data.

def make_messy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "emp_id":   [101, 102, 103, 104, 105, 106, 107, 102, 108],
        "dept":     ["eng", "Sales", "ENG", None, "sales", "HR", "hr", "Sales", "Finance"],
        "age":      [34.0, None, 28.0, 41.0, None, 55.0, 29.0, None, 38.0],
        "salary":   [88_000, 52_000, None, 91_000, 47_000, 61_000, 79_000, 52_000, 250_000],
        "score":    [4.2, 3.8, 4.5, None, 2.1, None, 3.9, 3.8, 4.0],
        "hire_year":[2018, 2020, 2019, 2017, None, 2015, 2021, 2020, 2022],
    })

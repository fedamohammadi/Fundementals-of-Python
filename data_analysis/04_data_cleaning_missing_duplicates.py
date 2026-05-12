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


# ==============================================================
# 1. Detecting Missing Values
# ==============================================================
# NaN (Not a Number) is pandas' sentinel for a missing numeric value;
# None fills the same role for object columns. Always diagnose
# missingness before touching anything else — the pattern of
# missing data determines which fix is appropriate.

def demo_detect_missing() -> None:
    df = make_messy_df()
    print(f"\n  Raw dataset:\n{df.to_string(index=False)}")

    # Boolean mask: True where a value is absent
    print(f"\n  df.isnull() (first 4 rows):\n{df.isnull().head(4)}")

    # Per-column count: fast summary of how much is missing
    null_counts = df.isnull().sum()
    null_pct    = (df.isnull().mean() * 100).round(1)
    missing_summary = pd.DataFrame({"count": null_counts, "pct": null_pct})
    missing_summary = missing_summary[missing_summary["count"] > 0]
    print(f"\n  Missing values per column:\n{missing_summary}")

    # Rows that contain at least one missing value
    incomplete_rows = df[df.isnull().any(axis=1)]
    print(f"\n  Rows with any missing value ({len(incomplete_rows)} of {len(df)}):")
    print(incomplete_rows.to_string(index=False))

    # Rows that are completely non-null
    complete_rows = df[df.notna().all(axis=1)]
    print(f"\n  Fully complete rows ({len(complete_rows)} of {len(df)}):")
    print(complete_rows.to_string(index=False))


# ==============================================================
# 2. Dropping Missing Values
# ==============================================================
# dropna removes rows or columns that exceed a missingness threshold.
# Use it only when the missing rows are few and not systematically
# different — otherwise you introduce selection bias.

def demo_drop_missing() -> None:
    df = make_messy_df()

    # Drop any row that has at least one NaN (default behaviour)
    df_any = df.dropna()
    print(f"\n  dropna() — drop rows with ANY null:")
    print(f"  {len(df)} rows -> {len(df_any)} rows")
    print(df_any.to_string(index=False))

    # Drop rows only when ALL values are NaN (rare but valid)
    df_all = df.dropna(how="all")
    print(f"\n  dropna(how='all') — drop rows where ALL values are null:")
    print(f"  {len(df)} rows -> {len(df_all)} rows  (none dropped here)")

    # thresh: keep rows that have at least k non-null values
    df_thresh = df.dropna(thresh=5)
    print(f"\n  dropna(thresh=5) — keep rows with >= 5 non-null values:")
    print(f"  {len(df)} rows -> {len(df_thresh)} rows")
    print(df_thresh.to_string(index=False))

    # subset: only consider certain columns when deciding to drop
    df_subset = df.dropna(subset=["salary", "dept"])
    print(f"\n  dropna(subset=['salary','dept']) — drop only if salary or dept is null:")
    print(f"  {len(df)} rows -> {len(df_subset)} rows")
    print(df_subset.to_string(index=False))

"""
Pandas Import and Export — CSV and Excel:
- Writing a CSV from scratch using the csv module
- Reading CSV with pandas: read_csv and key parameters
- Writing CSV with pandas: to_csv
- Reading and writing Excel: read_excel and to_excel
- Common read_csv parameters: sep, header, usecols, dtype, parse_dates
- Load-time filtering: skiprows, nrows, na_values
- Practical workflow: load → inspect → clean → export
"""

import csv
import io
import os
import tempfile
import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: employee records used across all sections
# ==============================================================

N_ROWS = 12


def make_employees() -> pd.DataFrame:
    rng    = np.random.default_rng(42)
    depts  = rng.choice(["Eng", "Sales", "HR", "Finance"], N_ROWS)
    years  = rng.integers(1, 15, N_ROWS).astype(float)
    salary = (40_000 + years * 3_500 + rng.normal(0, 5_000, N_ROWS)).round(0)
    rating = rng.choice(["good", "excellent", "needs_improvement"], N_ROWS)
    hired  = pd.date_range("2010-01-01", periods=N_ROWS, freq="3ME")
    return pd.DataFrame({
        "emp_id":    range(101, 101 + N_ROWS),
        "dept":      depts,
        "yrs_exp":   years,
        "salary":    salary,
        "rating":    rating,
        "hire_date": hired,
    })


# ==============================================================
# 1. Writing a CSV from Scratch (csv module)
# ==============================================================
# The stdlib csv module shows what a CSV file actually is: plain
# delimited text. Understanding the raw format prevents confusion
# when pandas read_csv behaves unexpectedly due to quoting or encoding.

def demo_write_csv_stdlib(path: str) -> None:
    rows = [
        ["id", "city",         "population"],
        [1,    "New York",     8_336_817],
        [2,    "Los Angeles",  3_979_576],
        [3,    "Chicago",      2_693_976],
        [4,    "Houston",      2_320_268],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n  Wrote CSV to: {path}")
    print(f"\n  Raw file content:")
    with open(path, encoding="utf-8") as f:
        for line in f:
            print(f"    {line}", end="")


# ==============================================================
# 2. Reading CSV with pandas
# ==============================================================
# pd.read_csv is the entry point for almost all tabular data.
# The defaults handle the common case; learning the key parameters
# saves hours debugging real-world files.

def demo_read_csv(path: str) -> None:
    # Basic read — pandas infers dtypes automatically
    df = pd.read_csv(path)
    print(f"\n  pd.read_csv result:")
    print(df.to_string(index=False))
    print(f"\n  dtypes:\n{df.dtypes}")
    print(f"\n  shape: {df.shape}")

    # Use a column as the index at read time
    df_idx = pd.read_csv(path, index_col="id")
    print(f"\n  index_col='id':")
    print(df_idx.to_string())

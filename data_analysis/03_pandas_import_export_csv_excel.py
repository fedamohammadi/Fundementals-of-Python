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

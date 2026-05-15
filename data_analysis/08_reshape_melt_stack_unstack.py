"""
Reshape, Melt, Stack, and Unstack:
- Wide vs. long format: when each form is appropriate
- melt(): converting wide columns into rows
- pivot(): collapsing long rows back into wide columns
- stack(): folding column levels into a row MultiIndex
- unstack(): rotating a row level into column labels
- crosstab(): computing frequency tables
- Practical example: reshaping a survey results dataset
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: quarterly sales in wide format
# ==============================================================
# Each row is a sales rep; columns Q1–Q4 hold their revenue.
# Wide format is compact and easy to read but awkward to filter
# or group by quarter — motivates the reshape operations below.

def make_wide_df() -> pd.DataFrame:
    return pd.DataFrame({
        "rep":    ["Ana", "Ben", "Carla", "Dan", "Eve"],
        "region": ["North", "South", "East", "North", "West"],
        "Q1":     [12_000, 8_500,  9_300,  10_500, 6_200],
        "Q2":     [14_200, 7_800,  11_000, 11_800, 9_800],
        "Q3":     [13_100, 9_200,  10_400, 12_600, 8_500],
        "Q4":     [15_800, 11_000, 13_700, 14_200, 10_300],
    })

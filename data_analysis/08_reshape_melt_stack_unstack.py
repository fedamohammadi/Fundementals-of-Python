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


# ==============================================================
# 1. Wide vs. Long Format
# ==============================================================
# Wide format: one row per entity, measurements spread across
# columns. Long (tidy) format: one row per observation, with a
# key column and a value column. Most pandas operations — groupby,
# plotting, and pivot_table — prefer long format.

def demo_wide_vs_long() -> None:
    wide = make_wide_df()
    print(f"\n  Wide format ({wide.shape[0]} rows × {wide.shape[1]} cols):")
    print(wide.to_string(index=False))

    # Manually built long example for Ana to illustrate the concept
    long_example = pd.DataFrame({
        "rep":     ["Ana"] * 4,
        "quarter": ["Q1", "Q2", "Q3", "Q4"],
        "revenue": [12_000, 14_200, 13_100, 15_800],
    })
    print(f"\n  Long format for Ana (4 rows × 3 cols):")
    print(long_example.to_string(index=False))

    print(f"\n  Wide: easy to scan. Long: easy to groupby and plot.")
    print(f"  groupby('quarter') on long — total per quarter:")
    totals = long_example.groupby("quarter")["revenue"].sum()
    print(totals.to_string())


# ==============================================================
# 2. melt(): Wide to Long
# ==============================================================
# melt() unpivots selected value columns into a single column,
# creating a key column for the former column names. id_vars=
# identifies columns that stay fixed. Use var_name= and value_name=
# to control the labels of the two new columns produced.

def demo_melt() -> None:
    wide = make_wide_df()

    # All four quarter columns become rows
    long = wide.melt(
        id_vars=["rep", "region"],
        value_vars=["Q1", "Q2", "Q3", "Q4"],
        var_name="quarter",
        value_name="revenue",
    )
    long = long.sort_values(["rep", "quarter"]).reset_index(drop=True)
    print(f"\n  melt() result ({long.shape[0]} rows × {long.shape[1]} cols):")
    print(long.to_string(index=False))

    # groupby now works naturally on the quarter column
    q_totals = long.groupby("quarter")["revenue"].sum()
    print(f"\n  Total revenue by quarter (from long format):")
    print(q_totals.to_string())

    # Melt only H1 quarters to compare halves
    h1 = wide.melt(id_vars=["rep"], value_vars=["Q1", "Q2"], var_name="half", value_name="revenue")
    print(f"\n  Melt of Q1/Q2 only (H1 comparison):")
    print(h1.to_string(index=False))

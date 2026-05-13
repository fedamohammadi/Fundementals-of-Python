"""
GroupBy, Aggregate, and Pivot:
- GroupBy basics: single key, multiple keys, group iteration
- Aggregation: built-in functions, agg() with multiple functions
- Custom aggregation functions and named aggregation
- transform: broadcasting group results back to the original shape
- filter: keeping or discarding entire groups by a condition
- Pivot tables: pd.pivot_table, margins, and fill values
- Practical example: regional sales dashboard
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: sales transactions
# ==============================================================
# Each row is one transaction: region, rep name, product,
# quarter, and revenue. Used across all sections.

def make_sales_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "region":  ["North", "South", "North", "East",  "South", "East",
                    "North", "West",  "West",  "South", "East",  "North"],
        "rep":     ["Ana",   "Ben",   "Ana",   "Carla", "Ben",   "Carla",
                    "Dan",   "Eve",   "Eve",   "Dan",   "Ana",   "Dan"],
        "product": ["Widget", "Gadget", "Widget", "Gadget", "Widget", "Widget",
                    "Gadget", "Widget", "Gadget", "Gadget", "Widget", "Widget"],
        "quarter": ["Q1", "Q1", "Q2", "Q1", "Q2", "Q2",
                    "Q1", "Q1", "Q2", "Q1", "Q2", "Q2"],
        "revenue": [12_000, 8_500, 14_200, 9_300, 7_800, 11_000,
                    10_500, 6_200, 9_800, 8_100, 13_400, 11_800],
        "units":   [40, 28, 48, 31, 26, 37, 35, 21, 33, 27, 45, 39],
    })

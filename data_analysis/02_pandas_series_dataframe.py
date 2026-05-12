"""
Pandas Series and DataFrame:
- Creating a Series (from list, dict, scalar)
- Creating a DataFrame (from dict, list of dicts, NumPy array)
- Basic inspection: head, tail, shape, dtypes, describe, info
- Indexing and selection: [], .loc, .iloc, boolean masks
- Adding and dropping columns; renaming
- Applying functions: apply, map, vectorized string ops
- Practical example: student gradebook analysis
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# 1. Creating a Series
# ==============================================================
# A Series is a 1-D labeled array. The index is what sets it apart
# from a plain NumPy array: each element has a name, enabling
# label-based alignment instead of position-based alignment.

def demo_series() -> None:
    # From a Python list — index defaults to 0, 1, 2, ...
    prices = pd.Series([102.5, 98.3, 105.7, 101.2, 99.8])
    print(f"\n  Series from list:\n{prices}")

    # Custom index makes the label explicit
    gdp = pd.Series(
        [21.43, 14.34, 5.81, 4.26, 3.09],
        index=["USA", "CHN", "JPN", "DEU", "IND"],
        name="GDP (trillion USD, 2020)",
    )
    print(f"\n  GDP Series (custom index):\n{gdp}")

    # Access by label or by integer position
    print(f"\n  gdp['USA']  = {gdp['USA']}")
    print(f"  gdp.iloc[1] = {gdp.iloc[1]}")

    # Boolean selection
    large = gdp[gdp > 5]
    print(f"\n  Economies with GDP > 5 trillion:\n{large}")

    # From a dict — keys become the index automatically
    unemployment = pd.Series(
        {"2019": 3.7, "2020": 8.1, "2021": 5.4, "2022": 3.6},
        name="US unemployment (%)",
    )
    print(f"\n  {unemployment.name}:\n{unemployment}")
    print(f"\n  mean={unemployment.mean():.2f}%  max={unemployment.max():.2f}%")


# ==============================================================
# 2. Creating a DataFrame
# ==============================================================
# A DataFrame is a 2-D labeled table — a dict of Series that share
# the same index. Each column holds one dtype, so mixed-type
# datasets fit naturally without any special wrappers.

def demo_dataframe() -> None:
    # From a dict of lists — the most common pattern
    df = pd.DataFrame({
        "name":    ["Alice", "Bob", "Clara", "David", "Emma"],
        "age":     [24, 31, 28, 45, 22],
        "income":  [52_000, 71_000, 63_500, 95_000, 47_000],
        "college": [True, False, True, True, False],
    })
    print(f"\n  DataFrame from dict:\n{df}")

    # From a list of dicts — each dict is one row
    records = [
        {"country": "USA", "year": 2020, "gdp": 21.43},
        {"country": "CHN", "year": 2020, "gdp": 14.34},
        {"country": "JPN", "year": 2020, "gdp":  5.81},
    ]
    df_records = pd.DataFrame(records)
    print(f"\n  DataFrame from list of dicts:\n{df_records}")

    # From a NumPy array — column names must be supplied
    rng   = np.random.default_rng(42)
    data  = rng.normal(0, 1, size=(4, 3)).round(3)
    df_np = pd.DataFrame(data, columns=["X1", "X2", "X3"])
    print(f"\n  DataFrame from NumPy array:\n{df_np}")

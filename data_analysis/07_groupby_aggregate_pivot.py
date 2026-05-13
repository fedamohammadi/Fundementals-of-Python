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


# ==============================================================
# 1. GroupBy Basics
# ==============================================================
# groupby() splits the DataFrame into groups by unique key values.
# Nothing is computed until you call an aggregation — it is lazy.
# Iterate over groups to inspect them; .size() gives group counts.

def demo_groupby_basics() -> None:
    df = make_sales_df()

    # Single key: total revenue per region
    total_by_region = df.groupby("region")["revenue"].sum()
    print(f"\n  Total revenue by region:")
    print(total_by_region.to_string())

    # Multiple keys: revenue broken down by region and quarter
    total_by_rq = df.groupby(["region", "quarter"])["revenue"].sum()
    print(f"\n  Total revenue by region and quarter:")
    print(total_by_rq.to_string())

    # Group size: number of transactions per region
    print(f"\n  Transactions per region:")
    print(df.groupby("region").size().to_string())

    # Iterating over groups
    print(f"\n  Top-revenue deal per region:")
    for name, group in df.groupby("region"):
        top = group.loc[group["revenue"].idxmax()]
        print(f"  [{name}]  rep={top['rep']}  revenue={top['revenue']:,}")


# ==============================================================
# 2. Aggregation: agg() with Multiple Functions
# ==============================================================
# agg() applies several functions at once and returns a flat
# DataFrame. Named aggregation (col=(source, func)) lets you
# control output column names directly — no MultiIndex cleanup.

def demo_aggregation() -> None:
    df = make_sales_df()

    # Multiple built-in functions on a single column
    stats = df.groupby("region")["revenue"].agg(["sum", "mean", "min", "max", "count"])
    print(f"\n  Revenue stats by region:")
    print(stats.round(0).to_string())

    # Named aggregation
    named = df.groupby("region").agg(
        total_rev   =("revenue", "sum"),
        avg_rev     =("revenue", "mean"),
        total_units =("units",   "sum"),
        n_deals     =("revenue", "count"),
    ).round(0)
    print(f"\n  Named aggregation by region:")
    print(named.to_string())

    # Different functions per column
    multi_col = df.groupby("quarter").agg({"revenue": "sum", "units": "mean"}).round(1)
    print(f"\n  Revenue sum and mean units per quarter:")
    print(multi_col.to_string())


# ==============================================================
# 3. Custom Aggregation Functions
# ==============================================================
# Pass any callable to agg(). The function receives a group's
# Series and must return a scalar. Use a lambda for short logic
# or a named function when it is reusable across aggregations.

def demo_custom_agg() -> None:
    df = make_sales_df()

    # Range (max - min) as a custom aggregation
    rev_range = df.groupby("region")["revenue"].agg(lambda s: s.max() - s.min())
    print(f"\n  Revenue range (max - min) by region:")
    print(rev_range.to_string())

    # Coefficient of variation: std / mean * 100
    def cv(s: pd.Series) -> float:
        return round(s.std() / s.mean() * 100, 1)

    print(f"\n  Revenue coefficient of variation (%) by region:")
    print(df.groupby("region")["revenue"].agg(cv).to_string())

    # Mix built-in and custom in one call
    mixed = df.groupby("product")["revenue"].agg(
        total =("sum"),
        spread=(lambda s: s.max() - s.min()),
    )
    print(f"\n  Revenue total and spread by product:")
    print(mixed.to_string())


# ==============================================================
# 4. transform: Broadcasting Results Back
# ==============================================================
# transform returns a Series aligned to the original DataFrame's
# index, repeating the group result for every row in the group.
# This is the right tool for adding group-level columns without
# collapsing the DataFrame.

def demo_transform() -> None:
    df = make_sales_df()

    # Add each row's regional average and deviation from it
    df["region_avg"] = df.groupby("region")["revenue"].transform("mean").round(0)
    df["vs_avg"]     = (df["revenue"] - df["region_avg"]).round(0)

    print(f"\n  Revenue vs. regional average:")
    print(df[["region", "rep", "revenue", "region_avg", "vs_avg"]].to_string(index=False))

    # Rank within each region (1 = highest revenue)
    df["rank_in_region"] = (
        df.groupby("region")["revenue"]
          .transform(lambda s: s.rank(ascending=False).astype(int))
    )
    print(f"\n  Rank within region (1 = highest):")
    ranked = df[["region", "rep", "revenue", "rank_in_region"]].sort_values(["region", "rank_in_region"])
    print(ranked.to_string(index=False))

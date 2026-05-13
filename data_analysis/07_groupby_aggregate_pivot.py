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


# ==============================================================
# 5. filter: Keeping or Discarding Entire Groups
# ==============================================================
# filter() takes a function that receives each group DataFrame
# and returns True (keep) or False (discard). Unlike a row-level
# boolean mask, it acts on entire groups at once.

def demo_filter() -> None:
    df = make_sales_df()

    # Keep only regions with total revenue above 30,000
    high_rev = df.groupby("region").filter(lambda g: g["revenue"].sum() > 30_000)
    n_before = df["region"].nunique()
    n_after  = high_rev["region"].nunique()
    print(f"\n  Regions with total revenue > 30,000 ({n_before} -> {n_after} regions):")
    print(high_rev[["region", "rep", "revenue"]].to_string(index=False))

    # Keep only reps who made at least 2 deals
    active = df.groupby("rep").filter(lambda g: len(g) >= 2)
    print(f"\n  Reps with >= 2 deals ({df['rep'].nunique()} -> {active['rep'].nunique()} reps):")
    print(active[["rep", "region", "revenue"]].to_string(index=False))

    # Keep only products whose mean revenue exceeds a threshold
    consistent = df.groupby("product").filter(lambda g: g["revenue"].mean() > 10_000)
    print(f"\n  Products with mean revenue > 10,000:")
    print(consistent[["product", "rep", "revenue"]].to_string(index=False))


# ==============================================================
# 6. Pivot Tables
# ==============================================================
# pd.pivot_table reshapes data into a spreadsheet-style summary.
# values= is the column to aggregate, index= and columns= define
# the row/column labels, aggfunc= the function.
# margins=True adds grand row and column totals.

def demo_pivot_table() -> None:
    df = make_sales_df()

    # Revenue by region (rows) and quarter (columns)
    pivot = pd.pivot_table(
        df, values="revenue", index="region", columns="quarter",
        aggfunc="sum", fill_value=0,
    )
    print(f"\n  Revenue pivot: region × quarter:")
    print(pivot.to_string())

    # With grand totals
    pivot_totals = pd.pivot_table(
        df, values="revenue", index="region", columns="quarter",
        aggfunc="sum", fill_value=0, margins=True, margins_name="Total",
    )
    print(f"\n  Same pivot with totals (margins=True):")
    print(pivot_totals.to_string())

    # Multiple aggregation functions at once
    pivot_multi = pd.pivot_table(
        df, values="revenue", index="product", columns="region",
        aggfunc=["sum", "mean"], fill_value=0,
    ).round(0)
    print(f"\n  Revenue sum and mean by product × region:")
    print(pivot_multi.to_string())


# ==============================================================
# 7. Practical Example: Regional Sales Dashboard
# ==============================================================
# Combine groupby and pivot to produce a one-page summary:
# a rep leaderboard, quarterly trend by region, and product mix.

def demo_sales_dashboard() -> None:
    df = make_sales_df()

    # Rep leaderboard
    leaderboard = (
        df.groupby(["rep", "region"])
          .agg(total_rev=("revenue", "sum"), deals=("revenue", "count"))
          .sort_values("total_rev", ascending=False)
          .reset_index()
    )
    print(f"\n  --- Rep leaderboard ---")
    print(leaderboard.to_string(index=False))

    # Quarterly revenue trend by region
    q_trend = pd.pivot_table(
        df, values="revenue", index="region", columns="quarter",
        aggfunc="sum", fill_value=0, margins=True, margins_name="Total",
    )
    print(f"\n  --- Quarterly revenue by region ---")
    print(q_trend.to_string())

    # Product mix: share of units sold per region
    product_mix = pd.pivot_table(
        df, values="units", index="region", columns="product",
        aggfunc="sum", fill_value=0,
    )
    product_mix_pct = product_mix.div(product_mix.sum(axis=1), axis=0).mul(100).round(1)
    print(f"\n  --- Product mix (% of units) by region ---")
    print(product_mix_pct.to_string())

    top_region = leaderboard.groupby("region")["total_rev"].sum().idxmax()
    print(f"\n  Top region by total revenue: {top_region}")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. GroupBy Basics")
    demo_groupby_basics()

    section("2. Aggregation: agg() with Multiple Functions")
    demo_aggregation()

    section("3. Custom Aggregation Functions")
    demo_custom_agg()

    section("4. transform: Broadcasting Results Back")
    demo_transform()

    section("5. filter: Keeping or Discarding Entire Groups")
    demo_filter()

    section("6. Pivot Tables")
    demo_pivot_table()

    section("7. Practical Example: Regional Sales Dashboard")
    demo_sales_dashboard()


if __name__ == "__main__":
    main()

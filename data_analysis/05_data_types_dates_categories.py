"""
Data Types, Dates, and Categories:
- Pandas data types: dtypes, info(), memory usage
- Type casting: astype(), pd.to_numeric(), pd.to_datetime()
- Parsing datetimes: string parsing, format strings, errors parameter
- Date arithmetic: timedelta, DateOffset, date_range
- Categorical data: pd.Categorical, CategoricalDtype, memory savings
- Ordered categories: sorting, comparison, and encoding
- Practical example: analysing an e-commerce orders dataset
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: e-commerce orders with mixed raw types
# ==============================================================
# The columns arrive as they would from a CSV: dates as strings,
# categories as freeform text, amounts as floats.

def make_orders_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":   [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "customer":   ["Alice", "Bob", "Alice", "Carol", "Bob", "Dave", "Alice", "Carol"],
        "category":   ["Electronics", "clothing", "Electronics", "Clothing",
                       "CLOTHING", "electronics", "Books", "books"],
        "amount":     [299.99, 45.00, 899.00, 120.00, 35.50, 650.00, 18.99, 22.50],
        "order_date": ["2024-01-15", "2024-01-20", "2024-02-03", "2024-02-14",
                       "2024-03-01", "2024-03-15", "2024-03-28", "2024-04-02"],
        "status":     ["shipped", "delivered", "shipped", "delivered",
                       "pending", "delivered", "shipped", "pending"],
    })


# ==============================================================
# 1. Pandas Data Types and dtypes
# ==============================================================
# Every Series has a dtype that controls memory layout and which
# operations are valid. Object dtype (used for strings) is the
# least efficient — a common source of hidden memory waste.
# df.info() gives a quick picture of types and null counts.

def demo_dtypes() -> None:
    df = make_orders_df()

    print(f"\n  DataFrame dtypes:")
    print(df.dtypes.to_string())

    print(f"\n  Memory usage per column (bytes):")
    print(df.memory_usage(deep=True).to_string())

    total_kb = df.memory_usage(deep=True).sum() / 1024
    print(f"\n  Total memory: {total_kb:.1f} KB")

    print(f"\n  df.info():")
    df.info()


# ==============================================================
# 2. Type Casting with astype()
# ==============================================================
# astype() converts a Series to a new dtype in one call.
# pd.to_numeric() and pd.to_datetime() are safer for data from
# external sources because they accept an errors= parameter.

def demo_type_casting() -> None:
    df = make_orders_df()

    # Cast amount to int (truncates decimal)
    df["amount_int"] = df["amount"].astype(int)
    print(f"\n  amount as float: {df['amount'].tolist()}")
    print(f"  amount as int  : {df['amount_int'].tolist()}")

    # pd.to_numeric with coerce: bad strings become NaN instead of raising
    messy = pd.Series(["10", "20", "abc", "40", None])
    numeric = pd.to_numeric(messy, errors="coerce")
    print(f"\n  pd.to_numeric(errors='coerce'):")
    print(f"  Input : {messy.tolist()}")
    print(f"  Output: {numeric.tolist()}")

    # Cast object column to nullable integer (supports NaN)
    s = pd.Series([1, 2, None, 4], dtype="object")
    s_int = s.astype("Int64")
    print(f"\n  object -> Int64 (nullable integer):")
    print(f"  Before dtype: {s.dtype}  |  After dtype: {s_int.dtype}")
    print(f"  Values: {s_int.tolist()}")

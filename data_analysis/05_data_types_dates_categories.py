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


# ==============================================================
# 3. Parsing Datetimes: pd.to_datetime()
# ==============================================================
# Datetime objects unlock a rich .dt accessor for extracting year,
# month, day, weekday, hour, etc. Always parse date strings as
# early as possible — downstream operations are much cleaner.

def demo_datetime_parsing() -> None:
    df = make_orders_df()

    # Convert the string column to datetime64
    df["order_date"] = pd.to_datetime(df["order_date"])
    print(f"\n  order_date dtype after parsing: {df['order_date'].dtype}")
    print(df[["order_id", "order_date"]].to_string(index=False))

    # .dt accessor: extract components
    df["year"]    = df["order_date"].dt.year
    df["month"]   = df["order_date"].dt.month
    df["weekday"] = df["order_date"].dt.day_name()
    print(f"\n  Extracted date components:")
    print(df[["order_id", "order_date", "year", "month", "weekday"]].to_string(index=False))

    # Parsing a non-standard format
    custom = pd.Series(["15/01/2024", "20/01/2024", "03/02/2024"])
    parsed = pd.to_datetime(custom, format="%d/%m/%Y")
    print(f"\n  Custom format '%d/%m/%Y':")
    print(f"  Input : {custom.tolist()}")
    print(f"  Parsed: {[str(d.date()) for d in parsed]}")

    # errors='coerce' turns unparseable strings into NaT
    bad = pd.Series(["2024-01-15", "not-a-date", "2024-03-01"])
    coerced = pd.to_datetime(bad, errors="coerce")
    print(f"\n  errors='coerce' on bad dates: {coerced.tolist()}")


# ==============================================================
# 4. Date Arithmetic: timedelta and date_range
# ==============================================================
# Datetime series support direct arithmetic. Subtracting two
# datetime columns yields a Timedelta series. DateOffset lets
# you add calendar-aware intervals (months, business days).

def demo_date_arithmetic() -> None:
    df = make_orders_df()
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Days since the first order in the dataset
    earliest = df["order_date"].min()
    df["days_since_first"] = (df["order_date"] - earliest).dt.days
    print(f"\n  Days since first order (reference: {earliest.date()}):")
    print(df[["order_id", "order_date", "days_since_first"]].to_string(index=False))

    # Add a fixed timedelta: estimated delivery = order + 5 days
    df["est_delivery"] = df["order_date"] + pd.Timedelta(days=5)
    print(f"\n  Estimated delivery (order + 5 days):")
    print(df[["order_id", "order_date", "est_delivery"]].to_string(index=False))

    # pd.date_range: generate a sequence of dates
    weekly = pd.date_range(start="2024-01-01", periods=6, freq="W")
    print(f"\n  Weekly date range (6 periods):")
    print(f"  {[str(d.date()) for d in weekly]}")

    monthly = pd.date_range(start="2024-01-01", periods=6, freq="ME")
    print(f"\n  Month-end date range (6 periods):")
    print(f"  {[str(d.date()) for d in monthly]}")


# ==============================================================
# 5. Categorical Data
# ==============================================================
# Storing repeated string values as Categorical saves memory and
# speeds up groupby. The underlying encoding is integer codes with
# a separate categories array — like a lookup table.

def demo_categorical() -> None:
    df = make_orders_df()

    # Normalise category labels first
    df["category"] = df["category"].str.strip().str.title()

    # Convert to Categorical
    df["cat_cat"] = df["category"].astype("category")
    print(f"\n  category as object  : {df['category'].dtype}")
    print(f"  category as category: {df['cat_cat'].dtype}")
    print(f"  Categories: {df['cat_cat'].cat.categories.tolist()}")
    print(f"  Codes     : {df['cat_cat'].cat.codes.tolist()}")

    # Memory comparison
    mem_obj = df["category"].memory_usage(deep=True)
    mem_cat = df["cat_cat"].memory_usage(deep=True)
    print(f"\n  Memory — object: {mem_obj} bytes  |  category: {mem_cat} bytes")

    # status also benefits from categorical encoding
    df["status_cat"] = df["status"].astype("category")
    print(f"\n  status categories: {df['status_cat'].cat.categories.tolist()}")
    print(f"  status codes     : {df['status_cat'].cat.codes.tolist()}")


# ==============================================================
# 6. Ordered Categories
# ==============================================================
# An ordered Categorical has a defined ranking between levels.
# This enables < > comparisons and meaningful sorting — critical
# for columns like priority, satisfaction rating, or size (S/M/L/XL).

def demo_ordered_categories() -> None:
    df = make_orders_df()

    # Define status order: pending < shipped < delivered
    status_order = pd.CategoricalDtype(
        categories=["pending", "shipped", "delivered"],
        ordered=True,
    )
    df["status_ord"] = df["status"].astype(status_order)

    print(f"\n  status_ord dtype: {df['status_ord'].dtype}")
    print(f"  Categories (ordered): {df['status_ord'].cat.categories.tolist()}")

    # Comparison: which orders have reached or passed 'shipped'?
    advanced = df[df["status_ord"] >= "shipped"]
    print(f"\n  Orders at or past 'shipped':")
    print(advanced[["order_id", "customer", "status_ord"]].to_string(index=False))

    # Sorting respects the category order, not alphabetical order
    sorted_df = df[["order_id", "status_ord"]].sort_values("status_ord")
    print(f"\n  Sorted by status (pending -> shipped -> delivered):")
    print(sorted_df.to_string(index=False))

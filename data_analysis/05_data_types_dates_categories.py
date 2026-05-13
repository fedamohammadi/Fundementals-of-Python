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

"""
Merge, Join, and Concat:
- pd.concat: stacking DataFrames vertically and horizontally
- Inner join: merge on matching keys only
- Left and right joins: preserving rows from one side
- Outer join: keeping all rows from both DataFrames
- Merging on multiple keys: composite keys and mismatched names
- Joining on index: DataFrame.join() and merge with left_index/right_index
- Practical example: building a complete customer-order report
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared datasets: customers and orders tables
# ==============================================================
# Two related tables: customers holds profile data, orders holds
# transaction records. Not every customer has an order, and not
# every order maps to a customer — by design, to show join gaps.

def make_customers() -> pd.DataFrame:
    return pd.DataFrame({
        "cust_id": [1, 2, 3, 4, 5],
        "name":    ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "city":    ["London", "Paris", "London", "Berlin", "Paris"],
        "tier":    ["Gold", "Silver", "Gold", "Bronze", "Silver"],
    })


def make_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106],
        "cust_id":  [1, 2, 1, 3, 6, 2],     # cust_id 6 has no matching customer
        "product":  ["Laptop", "Phone", "Tablet", "Camera", "Monitor", "Keyboard"],
        "amount":   [999.00, 499.00, 349.00, 799.00, 299.00, 89.00],
    })

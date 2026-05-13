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


# ==============================================================
# 1. Concatenating DataFrames: pd.concat
# ==============================================================
# concat stacks DataFrames along an axis — axis=0 stacks rows
# (default), axis=1 stacks columns. Use ignore_index=True to
# reset the index, and keys= to label each source chunk.

def demo_concat() -> None:
    q1 = pd.DataFrame({"product": ["Laptop", "Phone"], "sales": [10, 25], "quarter": ["Q1", "Q1"]})
    q2 = pd.DataFrame({"product": ["Laptop", "Phone"], "sales": [14, 30], "quarter": ["Q2", "Q2"]})
    q3 = pd.DataFrame({"product": ["Laptop", "Phone"], "sales": [8,  22], "quarter": ["Q3", "Q3"]})

    # Stack rows vertically — the most common use case
    all_quarters = pd.concat([q1, q2, q3], ignore_index=True)
    print(f"\n  pd.concat([q1, q2, q3], ignore_index=True):")
    print(all_quarters.to_string(index=False))

    # keys= labels each chunk with its source in a MultiIndex
    labeled = pd.concat([q1, q2, q3], keys=["Q1", "Q2", "Q3"])
    print(f"\n  pd.concat with keys=:")
    print(labeled.to_string())

    # axis=1: stack columns side by side
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"y": [4, 5, 6]})
    side = pd.concat([a, b], axis=1)
    print(f"\n  pd.concat([a, b], axis=1):")
    print(side.to_string(index=False))

    # Concatenating with different columns: NaN fills the gaps
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"b": [5, 6], "c": [7, 8]})
    unaligned = pd.concat([df1, df2], ignore_index=True)
    print(f"\n  Concatenating DataFrames with different columns:")
    print(unaligned.to_string(index=False))


# ==============================================================
# 2. Inner Join
# ==============================================================
# An inner join (the default) keeps only rows where the key
# exists in both DataFrames. Rows with no match on either side
# are silently dropped — always check row counts after a merge.

def demo_inner_join() -> None:
    customers = make_customers()
    orders    = make_orders()

    print(f"\n  Customers ({len(customers)} rows):")
    print(customers.to_string(index=False))
    print(f"\n  Orders ({len(orders)} rows):")
    print(orders.to_string(index=False))

    merged = pd.merge(customers, orders, on="cust_id", how="inner")
    print(f"\n  INNER JOIN on cust_id ({len(merged)} rows — only matched keys):")
    print(merged.to_string(index=False))

    unmatched_customers = len(customers) - merged["cust_id"].nunique()
    unmatched_orders    = orders[~orders["cust_id"].isin(customers["cust_id"])].shape[0]
    print(f"\n  Customers dropped (no orders): {unmatched_customers}")
    print(f"  Orders dropped (unknown customer): {unmatched_orders}")


# ==============================================================
# 3. Left and Right Joins
# ==============================================================
# A left join keeps all rows from the left DataFrame and fills
# NaN for columns from the right where no match exists.
# A right join is the mirror image. Both are useful when the
# "primary" table must stay intact (e.g. all customers).

def demo_left_right_join() -> None:
    customers = make_customers()
    orders    = make_orders()

    # Left join: every customer appears, even without orders
    left = pd.merge(customers, orders, on="cust_id", how="left")
    print(f"\n  LEFT JOIN — all customers, NaN where no order ({len(left)} rows):")
    print(left.to_string(index=False))

    # Right join: every order appears, even without a matching customer
    right = pd.merge(customers, orders, on="cust_id", how="right")
    print(f"\n  RIGHT JOIN — all orders, NaN where no customer ({len(right)} rows):")
    print(right.to_string(index=False))

    # Customers with no orders (NaN in order_id after left join)
    no_orders = left[left["order_id"].isna()]
    print(f"\n  Customers with no orders:")
    print(no_orders[["cust_id", "name", "city", "tier"]].to_string(index=False))


# ==============================================================
# 4. Outer Join
# ==============================================================
# An outer (full) join keeps all rows from both DataFrames and
# fills NaN wherever a match is missing. The indicator= parameter
# adds a _merge column that shows the origin of each row.

def demo_outer_join() -> None:
    customers = make_customers()
    orders    = make_orders()

    outer = pd.merge(customers, orders, on="cust_id", how="outer", indicator=True)
    print(f"\n  OUTER JOIN with indicator= ({len(outer)} rows):")
    print(outer.to_string(index=False))

    print(f"\n  Row origin counts:")
    print(outer["_merge"].value_counts().to_string())


# ==============================================================
# 5. Merging on Multiple Keys
# ==============================================================
# Use on=[...] with a list of columns to match rows on a
# composite key. When column names differ across tables use
# left_on= and right_on= instead of on=.

def demo_multi_key_merge() -> None:
    inventory = pd.DataFrame({
        "warehouse": ["A", "A", "B", "B"],
        "product":   ["Laptop", "Phone", "Laptop", "Phone"],
        "stock":     [50, 120, 30, 80],
    })
    shipments = pd.DataFrame({
        "warehouse": ["A", "A", "B", "C"],
        "product":   ["Laptop", "Phone", "Laptop", "Phone"],
        "shipped":   [10, 25, 5, 15],
    })

    print(f"\n  Inventory:\n{inventory.to_string(index=False)}")
    print(f"\n  Shipments:\n{shipments.to_string(index=False)}")

    merged = pd.merge(inventory, shipments, on=["warehouse", "product"], how="left")
    print(f"\n  Merged on ['warehouse', 'product'] (LEFT JOIN):")
    print(merged.to_string(index=False))

    # Mismatched column names: use left_on and right_on
    orders2 = pd.DataFrame({"id": [101, 102], "customer_id": [1, 2], "amount": [500, 200]})
    cust2   = pd.DataFrame({"cust_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
    joined  = pd.merge(orders2, cust2, left_on="customer_id", right_on="cust_id")
    print(f"\n  Merge with left_on='customer_id', right_on='cust_id':")
    print(joined.to_string(index=False))


# ==============================================================
# 6. Joining on Index
# ==============================================================
# DataFrame.join() merges on the index by default, which is
# convenient when your keys are already the index. merge() can
# also join on index via left_index / right_index flags.

def demo_index_join() -> None:
    customers = make_customers().set_index("cust_id")
    orders    = make_orders()

    print(f"\n  Customers indexed by cust_id:")
    print(customers.to_string())

    # join() uses the left DataFrame's index by default
    result = customers.join(orders.set_index("cust_id"), how="inner")
    print(f"\n  customers.join(orders, how='inner'):")
    print(result.to_string())

    # merge() equivalent using left_index / right_index
    merged = pd.merge(
        customers, orders,
        left_index=True, right_on="cust_id", how="inner",
    )
    print(f"\n  merge(left_index=True, right_on='cust_id'):")
    print(merged.to_string(index=False))

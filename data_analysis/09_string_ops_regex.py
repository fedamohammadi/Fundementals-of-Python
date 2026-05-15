"""
String Operations and Regular Expressions:
- str accessor: basic cleaning (strip, lower, upper, replace)
- Splitting and joining: str.split(), str.join()
- Filtering with str.contains(), str.startswith(), str.endswith()
- Extracting with str.extract() and str.extractall()
- str.findall(): collecting all pattern matches as lists
- Replacing with regex: str.replace() and the re module
- Practical example: parsing and cleaning a product catalog
"""

import re
import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: messy product catalog
# ==============================================================
# Names, categories, and SKUs arrive in raw form: inconsistent
# casing, extra whitespace, dollar signs embedded in price strings,
# and structured codes mixed into a single field. A realistic
# representation of data imported from a spreadsheet or web scrape.

def make_catalog_df() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": range(1, 9),
        "raw_name": [
            "  Widget Pro X1 ",
            "gadget LITE-200",
            "SUPER Widget   ",
            "gadget PRO-300  ",
            "  widget BASIC ",
            "Monitor 27-inch HD",
            "  keyboard WIRELESS-BT ",
            "Mouse  USB   ERGONOMIC",
        ],
        "category": [
            "electronics ", "Electronics", " ELECTRONICS",
            "electronics", "ELECTRONICS ", "electronics",
            "  Electronics", "electronics ",
        ],
        "sku": [
            "WDG-X1-2024", "GDG-L2-2023", "WDG-S3-2024",
            "GDG-P3-2023", "WDG-B1-2022", "MON-H27-2024",
            "KBD-WBT-2023", "MOU-UE-2024",
        ],
        "price_raw": [
            "$29.99", "$14.50", "$49.00", "$89.95",
            "$9.99", "$199.00", "$45.00", "$25.50",
        ],
    })


# ==============================================================
# 1. str Accessor: Basic Cleaning
# ==============================================================
# The .str accessor exposes string methods as vectorised operations
# over a whole Series. strip() removes leading/trailing whitespace,
# lower()/upper()/title() normalise case, and replace() does literal
# or regex substitution. Chain calls to build a cleaning pipeline.

def demo_str_basics() -> None:
    df = make_catalog_df()

    # Normalise category: strip whitespace then apply title case
    df["category_clean"] = df["category"].str.strip().str.title()
    print(f"\n  Raw category values:")
    print(df["category"].tolist())
    print(f"\n  Cleaned (strip + title):")
    print(df["category_clean"].tolist())

    # Normalise product names in the same way
    df["name_clean"] = df["raw_name"].str.strip().str.title()
    print(f"\n  Cleaned product names:")
    print(df["name_clean"].tolist())

    # Compare string lengths before and after stripping
    df["len_raw"]   = df["raw_name"].str.len()
    df["len_clean"] = df["name_clean"].str.len()
    print(f"\n  Length comparison (raw vs cleaned):")
    print(df[["raw_name", "len_raw", "name_clean", "len_clean"]].to_string(index=False))

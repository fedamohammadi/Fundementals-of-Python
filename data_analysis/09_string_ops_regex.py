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

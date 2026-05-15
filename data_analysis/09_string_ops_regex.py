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


# ==============================================================
# 2. Splitting and Joining
# ==============================================================
# str.split() splits each string on a delimiter and returns a
# Series of lists. expand=True returns a proper DataFrame of
# columns. str[-1] slices the last list element across the whole
# Series. str.join() is the reverse: it joins list items.

def demo_split_join() -> None:
    df = make_catalog_df()

    # Split SKU on '-': "WDG-X1-2024" -> ["WDG", "X1", "2024"]
    sku_parts = df["sku"].str.split("-", expand=True)
    sku_parts.columns = ["prefix", "code", "year"]
    print(f"\n  SKU split into parts:")
    print(sku_parts.to_string(index=False))

    # Extract just the year from the last element
    df["sku_year"] = df["sku"].str.split("-").str[-1].astype(int)
    print(f"\n  SKU year extracted:")
    print(df[["sku", "sku_year"]].to_string(index=False))

    # Brand = first word of the product name
    df["brand"] = df["raw_name"].str.strip().str.split().str[0].str.title()
    print(f"\n  Brand (first word of name):")
    print(df[["raw_name", "brand"]].to_string(index=False))

    # str.join() reassembles a list column into a single string
    parts = pd.Series([["WDG", "X1", "2024"], ["GDG", "L2", "2023"]])
    print(f"\n  str.join('-') on list column:")
    print(parts.str.join("-").tolist())


# ==============================================================
# 3. Filtering with str.contains, startswith, endswith
# ==============================================================
# str.contains() accepts a regex and returns a boolean mask; use
# case=False for case-insensitive matches and na=False so NaN rows
# evaluate to False instead of NaN. startswith / endswith do
# fast literal prefix / suffix checks without regex overhead.

def demo_filtering() -> None:
    df = make_catalog_df()

    # Products whose raw name contains "widget" (any case)
    widgets = df[df["raw_name"].str.contains("widget", case=False, na=False)]
    print(f"\n  Products containing 'widget' (case-insensitive):")
    print(widgets[["product_id", "raw_name"]].to_string(index=False))

    # SKUs that belong to the Widget product line
    wdg_skus = df[df["sku"].str.startswith("WDG")]
    print(f"\n  SKUs starting with 'WDG':")
    print(wdg_skus[["product_id", "sku"]].to_string(index=False))

    # Products released in 2024
    year_2024 = df[df["sku"].str.endswith("2024")]
    print(f"\n  SKUs ending with '2024':")
    print(year_2024[["product_id", "sku"]].to_string(index=False))

    # Combine conditions: gadgets from 2023
    gadget_2023 = df[df["sku"].str.startswith("GDG") & df["sku"].str.endswith("2023")]
    print(f"\n  Gadgets from 2023:")
    print(gadget_2023[["product_id", "sku", "raw_name"]].to_string(index=False))


# ==============================================================
# 4. Extracting with str.extract()
# ==============================================================
# str.extract() applies a regex with named capture groups and
# returns each group as a DataFrame column — only the first match
# per string. str.extractall() returns every match across all
# strings as a MultiIndex DataFrame, one row per match.

def demo_extract() -> None:
    df = make_catalog_df()

    # Named groups pull SKU parts into separate columns at once
    sku_parts = df["sku"].str.extract(
        r"(?P<prefix>[A-Z]+)-(?P<code>[A-Z0-9]+)-(?P<year>\d{4})"
    )
    print(f"\n  str.extract() named groups from SKU:")
    print(sku_parts.to_string(index=False))

    # Strip the dollar sign and parse price as a float
    df["price"] = df["price_raw"].str.extract(r"\$(\d+\.\d+)").astype(float)
    print(f"\n  Price extracted from raw string:")
    print(df[["price_raw", "price"]].to_string(index=False))

    # extractall: find every run of two or more uppercase letters
    caps = df["raw_name"].str.extractall(r"([A-Z]{2,})")
    caps.columns = ["all_caps_word"]
    print(f"\n  str.extractall() — all-caps tokens in raw names:")
    print(caps.to_string())


# ==============================================================
# 5. str.findall(): All Matches as Lists
# ==============================================================
# str.findall() returns a list of all non-overlapping matches for
# a pattern in each string. The result is a Series of lists, not
# a flat Series — use str.len() on it to count matches, or explode()
# to flatten into individual rows for further analysis.

def demo_findall() -> None:
    df = make_catalog_df()

    # Find all uppercase tokens in the raw product name
    df["caps_words"] = df["raw_name"].str.findall(r"[A-Z][A-Z0-9]+")
    print(f"\n  str.findall() — uppercase tokens per product:")
    print(df[["raw_name", "caps_words"]].to_string(index=False))

    # Count uppercase tokens per name
    df["n_caps"] = df["caps_words"].str.len()
    print(f"\n  Count of uppercase tokens per product:")
    print(df[["raw_name", "n_caps"]].to_string(index=False))

    # Find all digit sequences in each SKU
    df["digits"] = df["sku"].str.findall(r"\d+")
    print(f"\n  Digit sequences in SKU:")
    print(df[["sku", "digits"]].to_string(index=False))

    # Flatten all caps words into individual rows for frequency count
    all_tokens = df["caps_words"].explode().dropna()
    print(f"\n  Most common uppercase tokens across all names:")
    print(all_tokens.value_counts().head(6).to_string())


# ==============================================================
# 6. Replacing with Regex
# ==============================================================
# str.replace() accepts regex patterns by default (regex=True).
# Backreferences like r"\1" reuse captured groups to reorder or
# reformat parts of the match. The re module handles cases that
# need flags (re.IGNORECASE, re.MULTILINE) or pre-compiled patterns.

def demo_regex_replace() -> None:
    df = make_catalog_df()

    # Collapse two or more consecutive spaces into one
    df["name_clean"] = df["raw_name"].str.strip().str.replace(r"\s{2,}", " ", regex=True)
    print(f"\n  Collapsed extra spaces:")
    print(df[["raw_name", "name_clean"]].to_string(index=False))

    # Strip any character that is not a letter, digit, or hyphen from SKU
    df["sku_clean"] = df["sku"].str.replace(r"[^A-Z0-9\-]", "", regex=True)
    print(f"\n  SKU cleaned (non-alphanumeric removed):")
    print(df[["sku", "sku_clean"]].to_string(index=False))

    # Reformat SKU: move the year to the front using capture groups
    df["sku_reformatted"] = df["sku"].str.replace(
        r"^([A-Z]+)-([A-Z0-9]+)-(\d{4})$", r"\3-\1-\2", regex=True
    )
    print(f"\n  SKU reformatted (year first):")
    print(df[["sku", "sku_reformatted"]].to_string(index=False))

    # re module: compile a pattern once for repeated use
    pattern = re.compile(r"\b\d{4}\b")
    sample  = ["WDG-X1-2024", "GDG-L2-2023", "no-year-here"]
    matches = [m.group() if (m := pattern.search(s)) else None for s in sample]
    print(f"\n  re.compile to extract 4-digit years:")
    for s, m in zip(sample, matches):
        print(f"  {s!r:30s} -> {m}")


# ==============================================================
# 7. Practical Example: Cleaning a Product Catalog
# ==============================================================
# Full pipeline: strip and normalise text, parse embedded price
# strings, split SKU fields into structured columns, map prefix
# codes to product types, then compute a per-type price summary.

def demo_catalog_cleaning() -> None:
    df = make_catalog_df()

    # Step 1: normalise free-text columns
    df["name"]     = df["raw_name"].str.strip().str.replace(r"\s{2,}", " ", regex=True).str.title()
    df["category"] = df["category"].str.strip().str.title()

    # Step 2: parse price from its raw dollar-sign format
    df["price"] = df["price_raw"].str.extract(r"\$(\d+\.\d+)").astype(float)

    # Step 3: split SKU into structured fields
    sku_parts        = df["sku"].str.split("-", expand=True)
    df["sku_prefix"] = sku_parts[0]
    df["sku_code"]   = sku_parts[1]
    df["sku_year"]   = sku_parts[2].astype(int)

    # Step 4: decode product type from SKU prefix
    prefix_map = {"WDG": "Widget", "GDG": "Gadget", "MON": "Monitor",
                  "KBD": "Keyboard", "MOU": "Mouse"}
    df["product_type"] = df["sku_prefix"].map(prefix_map)

    print(f"\n  --- Cleaned catalog ---")
    cols = ["product_id", "name", "category", "price", "sku", "sku_year", "product_type"]
    print(df[cols].to_string(index=False))

    # Average price per product type
    avg_price = df.groupby("product_type")["price"].mean().round(2).sort_values(ascending=False)
    print(f"\n  Average price by product type:")
    print(avg_price.to_string())

    # Products released in 2024
    recent = df[df["sku_year"] == 2024][["name", "sku", "price"]]
    print(f"\n  Products released in 2024:")
    print(recent.to_string(index=False))

    # Verify all categories were unified
    print(f"\n  Unique categories after cleaning: {df['category'].unique().tolist()}")
    print(f"  All consistent: {df['category'].nunique() == 1}")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. str Accessor: Basic Cleaning")
    demo_str_basics()

    section("2. Splitting and Joining")
    demo_split_join()

    section("3. Filtering with str.contains, startswith, endswith")
    demo_filtering()

    section("4. Extracting with str.extract()")
    demo_extract()

    section("5. str.findall(): All Matches as Lists")
    demo_findall()

    section("6. Replacing with Regex")
    demo_regex_replace()

    section("7. Practical Example: Cleaning a Product Catalog")
    demo_catalog_cleaning()


if __name__ == "__main__":
    main()

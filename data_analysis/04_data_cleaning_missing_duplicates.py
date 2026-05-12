"""
Data Cleaning — Missing Values and Duplicates:
- Detecting missing values: isnull, notna, per-column counts
- Dropping missing values: dropna with axis, how, thresh, subset
- Filling missing values: fillna, ffill, bfill, interpolate
- Detecting and removing duplicates: duplicated, drop_duplicates
- Fixing inconsistent data: string normalisation, categorical standardisation
- Detecting and handling outliers: IQR method and Z-score
- Practical example: cleaning a messy employee survey dataset
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: intentionally messy survey records
# ==============================================================
# This dataset is used across all sections so each demo builds
# on the same realistic context rather than inventing new data.

def make_messy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "emp_id":   [101, 102, 103, 104, 105, 106, 107, 102, 108],
        "dept":     ["eng", "Sales", "ENG", None, "sales", "HR", "hr", "Sales", "Finance"],
        "age":      [34.0, None, 28.0, 41.0, None, 55.0, 29.0, None, 38.0],
        "salary":   [88_000, 52_000, None, 91_000, 47_000, 61_000, 79_000, 52_000, 250_000],
        "score":    [4.2, 3.8, 4.5, None, 2.1, None, 3.9, 3.8, 4.0],
        "hire_year":[2018, 2020, 2019, 2017, None, 2015, 2021, 2020, 2022],
    })


# ==============================================================
# 1. Detecting Missing Values
# ==============================================================
# NaN (Not a Number) is pandas' sentinel for a missing numeric value;
# None fills the same role for object columns. Always diagnose
# missingness before touching anything else — the pattern of
# missing data determines which fix is appropriate.

def demo_detect_missing() -> None:
    df = make_messy_df()
    print(f"\n  Raw dataset:\n{df.to_string(index=False)}")

    # Boolean mask: True where a value is absent
    print(f"\n  df.isnull() (first 4 rows):\n{df.isnull().head(4)}")

    # Per-column count: fast summary of how much is missing
    null_counts = df.isnull().sum()
    null_pct    = (df.isnull().mean() * 100).round(1)
    missing_summary = pd.DataFrame({"count": null_counts, "pct": null_pct})
    missing_summary = missing_summary[missing_summary["count"] > 0]
    print(f"\n  Missing values per column:\n{missing_summary}")

    # Rows that contain at least one missing value
    incomplete_rows = df[df.isnull().any(axis=1)]
    print(f"\n  Rows with any missing value ({len(incomplete_rows)} of {len(df)}):")
    print(incomplete_rows.to_string(index=False))

    # Rows that are completely non-null
    complete_rows = df[df.notna().all(axis=1)]
    print(f"\n  Fully complete rows ({len(complete_rows)} of {len(df)}):")
    print(complete_rows.to_string(index=False))


# ==============================================================
# 2. Dropping Missing Values
# ==============================================================
# dropna removes rows or columns that exceed a missingness threshold.
# Use it only when the missing rows are few and not systematically
# different — otherwise you introduce selection bias.

def demo_drop_missing() -> None:
    df = make_messy_df()

    # Drop any row that has at least one NaN (default behaviour)
    df_any = df.dropna()
    print(f"\n  dropna() — drop rows with ANY null:")
    print(f"  {len(df)} rows -> {len(df_any)} rows")
    print(df_any.to_string(index=False))

    # Drop rows only when ALL values are NaN (rare but valid)
    df_all = df.dropna(how="all")
    print(f"\n  dropna(how='all') — drop rows where ALL values are null:")
    print(f"  {len(df)} rows -> {len(df_all)} rows  (none dropped here)")

    # thresh: keep rows that have at least k non-null values
    df_thresh = df.dropna(thresh=5)
    print(f"\n  dropna(thresh=5) — keep rows with >= 5 non-null values:")
    print(f"  {len(df)} rows -> {len(df_thresh)} rows")
    print(df_thresh.to_string(index=False))

    # subset: only consider certain columns when deciding to drop
    df_subset = df.dropna(subset=["salary", "dept"])
    print(f"\n  dropna(subset=['salary','dept']) — drop only if salary or dept is null:")
    print(f"  {len(df)} rows -> {len(df_subset)} rows")
    print(df_subset.to_string(index=False))


# ==============================================================
# 3. Filling Missing Values
# ==============================================================
# Filling (imputation) keeps all rows in the dataset — valuable
# when dropping would lose too much data. The right fill strategy
# depends on context: a constant, the column mean, or the
# neighbouring value in a time series.

def demo_fill_missing() -> None:
    df = make_messy_df()

    # Fill with a scalar constant
    df_const = df.copy()
    df_const["score"]     = df_const["score"].fillna(0.0)
    df_const["hire_year"] = df_const["hire_year"].fillna(9999)
    print(f"\n  fillna(constant) — score->0, hire_year->9999:")
    print(df_const[["emp_id", "score", "hire_year"]].to_string(index=False))

    # Fill with column statistics (mean for numeric columns)
    df_mean = df.copy()
    df_mean["age"]    = df_mean["age"].fillna(df_mean["age"].mean().round(1))
    df_mean["salary"] = df_mean["salary"].fillna(df_mean["salary"].median())
    df_mean["score"]  = df_mean["score"].fillna(df_mean["score"].mean().round(2))
    print(f"\n  fillna(mean/median):")
    print(df_mean[["emp_id", "age", "salary", "score"]].to_string(index=False))

    # Forward fill (ffill): carry the last valid value forward
    # Useful for time-ordered data where gaps should inherit the prior state.
    ts = pd.Series([10.0, None, None, 13.0, None, 15.0])
    print(f"\n  Forward fill (ffill) on a time series:")
    print(f"  Original : {ts.tolist()}")
    print(f"  After    : {ts.ffill().tolist()}")

    # Backward fill (bfill): fill from the next valid value
    print(f"\n  Backward fill (bfill):")
    print(f"  After    : {ts.bfill().tolist()}")

    # interpolate: estimate missing values using linear interpolation
    ts2 = pd.Series([1.0, None, None, 4.0, None, 10.0])
    print(f"\n  Linear interpolation:")
    print(f"  Original    : {ts2.tolist()}")
    print(f"  Interpolated: {ts2.interpolate().tolist()}")


# ==============================================================
# 4. Detecting and Removing Duplicates
# ==============================================================
# A duplicate row is an observation that appears more than once.
# duplicated() flags them; drop_duplicates() removes them.
# Choosing which copy to keep — first, last, or none — matters
# when the duplicates differ on other columns.

def demo_duplicates() -> None:
    df = make_messy_df()

    # Flag duplicate rows (default: all columns, keep first occurrence)
    dup_mask = df.duplicated()
    print(f"\n  df.duplicated() — is each row a duplicate?")
    print(f"  {dup_mask.values}")
    print(f"  Number of duplicate rows: {dup_mask.sum()}")

    # Show the duplicate rows
    print(f"\n  Duplicate rows:\n{df[dup_mask].to_string(index=False)}")

    # Remove duplicates, keeping the first occurrence
    df_dedup = df.drop_duplicates()
    print(f"\n  After drop_duplicates(): {len(df)} -> {len(df_dedup)} rows")

    # Duplicate on a subset of columns: same emp_id entered twice
    dup_id = df.duplicated(subset=["emp_id"])
    print(f"\n  Duplicates on emp_id only:")
    print(df[dup_id][["emp_id", "dept", "salary"]].to_string(index=False))

    # Keep the last occurrence (e.g. most recent record wins)
    df_last = df.drop_duplicates(subset=["emp_id"], keep="last")
    print(f"\n  drop_duplicates(subset=['emp_id'], keep='last'):")
    print(df_last[["emp_id", "dept", "salary"]].to_string(index=False))


# ==============================================================
# 5. Fixing Inconsistent Data
# ==============================================================
# Inconsistency hides in plain sight: "eng" and "ENG" and "Eng"
# all mean the same department, but groupby treats them as three.
# String normalisation and categorical mapping fix this before
# any aggregation or modelling step.

def demo_inconsistent_data() -> None:
    df = make_messy_df()

    print(f"\n  Raw dept values: {df['dept'].tolist()}")
    print(f"  Unique values  : {df['dept'].unique()}")

    # Step 1: strip whitespace and lowercase
    df["dept_clean"] = df["dept"].str.strip().str.lower()
    print(f"\n  After .str.strip().str.lower():")
    print(f"  {df['dept_clean'].tolist()}")

    # Step 2: unify aliases with a mapping dict
    dept_map = {
        "eng":     "Engineering",
        "sales":   "Sales",
        "hr":      "HR",
        "finance": "Finance",
    }
    df["dept_clean"] = df["dept_clean"].map(dept_map)
    print(f"\n  After map(dept_map):")
    print(f"  {df['dept_clean'].tolist()}")

    # Step 3: inspect the result
    print(f"\n  Value counts after cleaning:")
    print(df["dept_clean"].value_counts().to_string())

    # Fixing a numeric column with a bad sentinel value
    # hire_year == 9999 was used as a placeholder; replace with NaN
    df2 = df.copy()
    df2["hire_year"] = df2["hire_year"].replace(9999, np.nan)
    print(f"\n  Replace sentinel 9999 -> NaN in hire_year:")
    print(df2[["emp_id", "hire_year"]].to_string(index=False))

    # clip: enforce a valid range (age must be 18-70)
    ages = pd.Series([16.0, 25.0, 34.0, 72.0, 28.0])
    print(f"\n  Ages before clip   : {ages.tolist()}")
    print(f"  Ages after clip(18,70): {ages.clip(18, 70).tolist()}")


# ==============================================================
# 6. Detecting and Handling Outliers
# ==============================================================
# Outliers can be legitimate extremes or data-entry errors.
# Two common detection rules:
#   IQR method : flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
#   Z-score    : flag values where |z| > 3  (more than 3 std from mean)
# After detection, choose: drop, cap (winsorise), or investigate.

def demo_outliers() -> None:
    df = make_messy_df()
    salaries = df["salary"].dropna()

    # ---- IQR method ----
    q1, q3 = salaries.quantile(0.25), salaries.quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outlier_mask = (salaries < lo) | (salaries > hi)
    print(f"\n  IQR method on salary:")
    print(f"    Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
    print(f"    Fence: [{lo:,.0f}, {hi:,.0f}]")
    print(f"    Outlier values: {salaries[outlier_mask].tolist()}")

    # ---- Z-score method ----
    z_scores = (salaries - salaries.mean()) / salaries.std(ddof=1)
    z_outliers = salaries[z_scores.abs() > 2.5]
    print(f"\n  Z-score method (|z| > 2.5):")
    print(f"    Mean={salaries.mean():,.0f}  Std={salaries.std(ddof=1):,.0f}")
    print(f"    Outlier values: {z_outliers.tolist()}")

    # ---- Winsorising (capping) ----
    # Replace outliers with the fence values rather than dropping them.
    salaries_capped = salaries.clip(lower=lo, upper=hi)
    print(f"\n  Original salaries : {salaries.tolist()}")
    print(f"  Winsorised (IQR)  : {salaries_capped.tolist()}")

    # ---- Dropping outliers ----
    df_clean = df.copy()
    df_clean = df_clean[
        (df_clean["salary"] >= lo) & (df_clean["salary"] <= hi)
        | df_clean["salary"].isna()
    ]
    print(f"\n  After dropping IQR outliers: {len(df)} -> {len(df_clean)} rows")


# ==============================================================
# 7. Practical Example: Cleaning a Messy Survey Dataset
# ==============================================================
# A full cleaning pipeline on a single DataFrame, applying every
# technique above in the order a real analyst would reach for them:
# inspect -> deduplicate -> standardise -> impute -> cap outliers.

def demo_cleaning_pipeline() -> None:
    df = make_messy_df()

    print(f"\n  --- Raw data ---")
    print(df.to_string(index=False))
    print(f"\n  Shape: {df.shape}  |  Nulls: {df.isnull().sum().sum()}")

    # Step 1: deduplicate on emp_id, keeping the first record
    df = df.drop_duplicates(subset=["emp_id"], keep="first")
    print(f"\n  Step 1 — Deduplication: {df.shape[0]} rows remain")

    # Step 2: standardise dept labels
    dept_map = {"eng": "Engineering", "sales": "Sales", "hr": "HR", "finance": "Finance"}
    df["dept"] = df["dept"].str.strip().str.lower().map(dept_map)
    print(f"\n  Step 2 — Dept after standardisation: {df['dept'].tolist()}")

    # Step 3: impute missing values
    df["age"]       = df["age"].fillna(df["age"].median())
    df["score"]     = df["score"].fillna(df["score"].mean().round(2))
    df["hire_year"] = df["hire_year"].fillna(df["hire_year"].median()).astype(int)
    df["salary"]    = df["salary"].fillna(df["salary"].median())
    print(f"\n  Step 3 — Nulls after imputation: {df.isnull().sum().sum()}")

    # Step 4: cap salary outliers using the IQR fence
    q1, q3 = df["salary"].quantile(0.25), df["salary"].quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["salary"] = df["salary"].clip(lower=lo, upper=hi)
    print(f"\n  Step 4 — Salary capped to [{lo:,.0f}, {hi:,.0f}]")

    # Step 5: add a derived tenure column
    df["tenure_yrs"] = 2024 - df["hire_year"]

    print(f"\n  --- Clean data ---")
    print(df.to_string(index=False))
    print(f"\n  Shape: {df.shape}  |  Nulls: {df.isnull().sum().sum()}")

    # Summary statistics on the clean data
    print(f"\n  Summary statistics (numeric columns):")
    print(df[["age", "salary", "score", "tenure_yrs"]].describe().round(2))


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Detecting Missing Values")
    demo_detect_missing()

    section("2. Dropping Missing Values")
    demo_drop_missing()

    section("3. Filling Missing Values")
    demo_fill_missing()

    section("4. Detecting and Removing Duplicates")
    demo_duplicates()

    section("5. Fixing Inconsistent Data")
    demo_inconsistent_data()

    section("6. Detecting and Handling Outliers")
    demo_outliers()

    section("7. Practical Example: Cleaning a Messy Survey Dataset")
    demo_cleaning_pipeline()


if __name__ == "__main__":
    main()

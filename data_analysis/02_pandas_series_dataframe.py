"""
Pandas Series and DataFrame:
- Creating a Series (from list, dict, scalar)
- Creating a DataFrame (from dict, list of dicts, NumPy array)
- Basic inspection: head, tail, shape, dtypes, describe, info
- Indexing and selection: [], .loc, .iloc, boolean masks
- Adding and dropping columns; renaming
- Applying functions: apply, map, vectorized string ops
- Practical example: student gradebook analysis
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# 1. Creating a Series
# ==============================================================
# A Series is a 1-D labeled array. The index is what sets it apart
# from a plain NumPy array: each element has a name, enabling
# label-based alignment instead of position-based alignment.

def demo_series() -> None:
    # From a Python list — index defaults to 0, 1, 2, ...
    prices = pd.Series([102.5, 98.3, 105.7, 101.2, 99.8])
    print(f"\n  Series from list:\n{prices}")

    # Custom index makes the label explicit
    gdp = pd.Series(
        [21.43, 14.34, 5.81, 4.26, 3.09],
        index=["USA", "CHN", "JPN", "DEU", "IND"],
        name="GDP (trillion USD, 2020)",
    )
    print(f"\n  GDP Series (custom index):\n{gdp}")

    # Access by label or by integer position
    print(f"\n  gdp['USA']  = {gdp['USA']}")
    print(f"  gdp.iloc[1] = {gdp.iloc[1]}")

    # Boolean selection
    large = gdp[gdp > 5]
    print(f"\n  Economies with GDP > 5 trillion:\n{large}")

    # From a dict — keys become the index automatically
    unemployment = pd.Series(
        {"2019": 3.7, "2020": 8.1, "2021": 5.4, "2022": 3.6},
        name="US unemployment (%)",
    )
    print(f"\n  {unemployment.name}:\n{unemployment}")
    print(f"\n  mean={unemployment.mean():.2f}%  max={unemployment.max():.2f}%")


# ==============================================================
# 2. Creating a DataFrame
# ==============================================================
# A DataFrame is a 2-D labeled table — a dict of Series that share
# the same index. Each column holds one dtype, so mixed-type
# datasets fit naturally without any special wrappers.

def demo_dataframe() -> None:
    # From a dict of lists — the most common pattern
    df = pd.DataFrame({
        "name":    ["Alice", "Bob", "Clara", "David", "Emma"],
        "age":     [24, 31, 28, 45, 22],
        "income":  [52_000, 71_000, 63_500, 95_000, 47_000],
        "college": [True, False, True, True, False],
    })
    print(f"\n  DataFrame from dict:\n{df}")

    # From a list of dicts — each dict is one row
    records = [
        {"country": "USA", "year": 2020, "gdp": 21.43},
        {"country": "CHN", "year": 2020, "gdp": 14.34},
        {"country": "JPN", "year": 2020, "gdp":  5.81},
    ]
    df_records = pd.DataFrame(records)
    print(f"\n  DataFrame from list of dicts:\n{df_records}")

    # From a NumPy array — column names must be supplied
    rng   = np.random.default_rng(42)
    data  = rng.normal(0, 1, size=(4, 3)).round(3)
    df_np = pd.DataFrame(data, columns=["X1", "X2", "X3"])
    print(f"\n  DataFrame from NumPy array:\n{df_np}")


# ==============================================================
# Shared dataset: employee records used by sections 3–6
# ==============================================================

def make_employees() -> pd.DataFrame:
    return pd.DataFrame({
        "emp_id":  [101, 102, 103, 104, 105, 106],
        "dept":    ["Sales", "Eng", "Eng", "HR", "Sales", "Eng"],
        "salary":  [52_000, 88_000, 91_000, 61_000, 55_000, 79_000],
        "yrs_exp": [2, 5, 7, 3, 4, 6],
        "remote":  [True, True, False, False, True, True],
    })


# ==============================================================
# 3. Basic Inspection
# ==============================================================
# Before any analysis, always inspect your data. These methods
# catch dimension mistakes, wrong dtypes, and missing values
# before they corrupt downstream calculations.

def demo_inspection() -> None:
    df = make_employees()

    print(f"\n  df.head(3):\n{df.head(3)}")
    print(f"\n  df.tail(2):\n{df.tail(2)}")
    print(f"\n  df.shape : {df.shape}   (rows, columns)")
    print(f"\n  df.dtypes:\n{df.dtypes}")
    print(f"\n  df.describe():\n{df.describe().round(2)}")
    print(f"\n  df.info():")
    df.info()


# ==============================================================
# 4. Indexing and Selection
# ==============================================================
# pandas has three selection systems:
#   []      — column by name, or rows by boolean mask
#   .loc[]  — label-based: rows and columns by label
#   .iloc[] — position-based: rows and columns by integer index
# Mixing these is the most common pandas beginner mistake.

def demo_indexing() -> None:
    df = make_employees()

    # Single column → Series
    print(f"\n  df['salary'] values:\n{df['salary'].values}")

    # Multiple columns → DataFrame
    print(f"\n  df[['emp_id','dept','salary']]:\n{df[['emp_id','dept','salary']]}")

    # Boolean filter
    engineers = df[df["dept"] == "Eng"]
    print(f"\n  Engineers only:\n{engineers}")

    # Compound boolean condition (parentheses required)
    high_remote = df[(df["salary"] > 70_000) & (df["remote"] == True)]
    print(f"\n  High earners who work remotely:")
    print(high_remote[["emp_id", "dept", "salary"]])

    # .loc — label-based (set a named index first)
    df_idx = df.set_index("emp_id")
    print(f"\n  .loc[103, ['dept','salary']]:\n{df_idx.loc[103, ['dept','salary']]}")

    # .iloc — purely positional
    print(f"\n  .iloc[0:3, 0:3] (top-left block):\n{df.iloc[0:3, 0:3]}")


# ==============================================================
# 5. Adding and Dropping Columns
# ==============================================================
# Derived columns are computed from existing ones and assigned back.
# Drop removes columns (axis=1) or rows (axis=0).
# Rename replaces column labels without touching values.

def demo_columns() -> None:
    df = make_employees()

    # Add derived columns
    df["monthly_salary"] = df["salary"] / 12
    df["seniority"] = df["yrs_exp"].apply(
        lambda y: "junior" if y < 4 else ("mid" if y < 7 else "senior")
    )
    print(f"\n  Derived columns added:")
    print(df[["emp_id", "salary", "monthly_salary", "seniority"]])

    # Drop a column
    df_slim = df.drop(columns=["monthly_salary"])
    print(f"\n  Columns after drop: {df_slim.columns.tolist()}")

    # Drop rows by index position
    df_trimmed = df.drop(index=[0, 5])
    print(f"\n  After dropping rows 0 and 5:")
    print(df_trimmed[["emp_id", "dept"]].to_string(index=False))

    # Rename columns
    df_renamed = df.rename(columns={"yrs_exp": "experience", "dept": "department"})
    print(f"\n  Renamed columns: {df_renamed.columns.tolist()}")


# ==============================================================
# 6. Applying Functions
# ==============================================================
# .apply() maps any callable over rows or columns.
# .map() replaces individual Series values via a dict or function.
# For simple arithmetic, vectorized operators are the fastest path.

def demo_apply() -> None:
    df = make_employees()

    # Vectorized arithmetic (NumPy speed, no Python loop)
    df["salary_k"] = df["salary"] / 1_000
    print(f"\n  Salary in thousands (vectorised): {df['salary_k'].values}")

    # apply on a column: custom bucketing
    def salary_band(s: float) -> str:
        if s < 60_000: return "low"
        if s < 80_000: return "mid"
        return "high"

    df["band"] = df["salary"].apply(salary_band)
    print(f"\n  Salary bands:\n{df[['emp_id', 'salary', 'band']]}")

    # apply on rows (axis=1): combine multiple columns
    df["score"] = df.apply(
        lambda row: row["salary"] / 1_000 + row["yrs_exp"] * 5,
        axis=1,
    )
    print(f"\n  Composite score (salary/1k + 5×experience):")
    print(df[["emp_id", "salary", "yrs_exp", "score"]])

    # map: recode categoricals with a lookup dict
    dept_map = {"Sales": "Revenue", "Eng": "Engineering", "HR": "People"}
    df["dept_full"] = df["dept"].map(dept_map)
    print(f"\n  Department full names:\n{df[['dept', 'dept_full']]}")

    # String accessor — .str exposes vectorized string methods
    df["dept_upper"] = df["dept"].str.upper()
    print(f"\n  .str.upper(): {df['dept_upper'].tolist()}")


# ==============================================================
# 7. Practical Example: Student Gradebook Analysis
# ==============================================================
# Compute weighted averages, assign letter grades, rank students,
# and summarise by section — a complete mini-pipeline that ties
# together creation, column ops, apply, and groupby.

def make_gradebook() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "student": [f"S{i:02d}" for i in range(1, 11)],
        "section": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "midterm": rng.integers(50, 100, 10).tolist(),
        "final":   rng.integers(50, 100, 10).tolist(),
        "hw_avg":  rng.integers(60, 100, 10).tolist(),
    })


def letter_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def demo_gradebook() -> None:
    df = make_gradebook()
    print(f"\n  Raw gradebook:\n{df.to_string(index=False)}")

    # Weighted average: midterm 30%, final 50%, hw 20%
    df["weighted_avg"] = (
        df["midterm"] * 0.30 +
        df["final"]   * 0.50 +
        df["hw_avg"]  * 0.20
    ).round(1)
    df["letter"] = df["weighted_avg"].apply(letter_grade)
    df["rank"]   = df["weighted_avg"].rank(ascending=False, method="min").astype(int)

    result_cols = ["student", "section", "weighted_avg", "letter", "rank"]
    print(f"\n  Grades and ranks:")
    print(df[result_cols].sort_values("rank").to_string(index=False))

    # Section summary using groupby + agg
    summary = (
        df.groupby("section")["weighted_avg"]
        .agg(count="count", mean="mean", std="std", min="min", max="max")
        .round(2)
    )
    print(f"\n  Summary by section:\n{summary}")

    # Grade distribution across both sections
    grade_counts = df["letter"].value_counts().sort_index()
    print(f"\n  Grade distribution:\n{grade_counts}")

    # Top 3 students overall
    top3 = df.nsmallest(3, "rank")[["student", "section", "weighted_avg", "letter"]]
    print(f"\n  Top 3 students:\n{top3.to_string(index=False)}")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Creating a Series")
    demo_series()

    section("2. Creating a DataFrame")
    demo_dataframe()

    section("3. Basic Inspection")
    demo_inspection()

    section("4. Indexing and Selection")
    demo_indexing()

    section("5. Adding and Dropping Columns")
    demo_columns()

    section("6. Applying Functions")
    demo_apply()

    section("7. Practical Example: Student Gradebook")
    demo_gradebook()


if __name__ == "__main__":
    main()

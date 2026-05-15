"""
Reshape, Melt, Stack, and Unstack:
- Wide vs. long format: when each form is appropriate
- melt(): converting wide columns into rows
- pivot(): collapsing long rows back into wide columns
- stack(): folding column levels into a row MultiIndex
- unstack(): rotating a row level into column labels
- crosstab(): computing frequency tables
- Practical example: reshaping a survey results dataset
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: quarterly sales in wide format
# ==============================================================
# Each row is a sales rep; columns Q1–Q4 hold their revenue.
# Wide format is compact and easy to read but awkward to filter
# or group by quarter — motivates the reshape operations below.

def make_wide_df() -> pd.DataFrame:
    return pd.DataFrame({
        "rep":    ["Ana", "Ben", "Carla", "Dan", "Eve"],
        "region": ["North", "South", "East", "North", "West"],
        "Q1":     [12_000, 8_500,  9_300,  10_500, 6_200],
        "Q2":     [14_200, 7_800,  11_000, 11_800, 9_800],
        "Q3":     [13_100, 9_200,  10_400, 12_600, 8_500],
        "Q4":     [15_800, 11_000, 13_700, 14_200, 10_300],
    })


# ==============================================================
# 1. Wide vs. Long Format
# ==============================================================
# Wide format: one row per entity, measurements spread across
# columns. Long (tidy) format: one row per observation, with a
# key column and a value column. Most pandas operations — groupby,
# plotting, and pivot_table — prefer long format.

def demo_wide_vs_long() -> None:
    wide = make_wide_df()
    print(f"\n  Wide format ({wide.shape[0]} rows × {wide.shape[1]} cols):")
    print(wide.to_string(index=False))

    # Manually built long example for Ana to illustrate the concept
    long_example = pd.DataFrame({
        "rep":     ["Ana"] * 4,
        "quarter": ["Q1", "Q2", "Q3", "Q4"],
        "revenue": [12_000, 14_200, 13_100, 15_800],
    })
    print(f"\n  Long format for Ana (4 rows × 3 cols):")
    print(long_example.to_string(index=False))

    print(f"\n  Wide: easy to scan. Long: easy to groupby and plot.")
    print(f"  groupby('quarter') on long — total per quarter:")
    totals = long_example.groupby("quarter")["revenue"].sum()
    print(totals.to_string())


# ==============================================================
# 2. melt(): Wide to Long
# ==============================================================
# melt() unpivots selected value columns into a single column,
# creating a key column for the former column names. id_vars=
# identifies columns that stay fixed. Use var_name= and value_name=
# to control the labels of the two new columns produced.

def demo_melt() -> None:
    wide = make_wide_df()

    # All four quarter columns become rows
    long = wide.melt(
        id_vars=["rep", "region"],
        value_vars=["Q1", "Q2", "Q3", "Q4"],
        var_name="quarter",
        value_name="revenue",
    )
    long = long.sort_values(["rep", "quarter"]).reset_index(drop=True)
    print(f"\n  melt() result ({long.shape[0]} rows × {long.shape[1]} cols):")
    print(long.to_string(index=False))

    # groupby now works naturally on the quarter column
    q_totals = long.groupby("quarter")["revenue"].sum()
    print(f"\n  Total revenue by quarter (from long format):")
    print(q_totals.to_string())

    # Melt only H1 quarters to compare halves
    h1 = wide.melt(id_vars=["rep"], value_vars=["Q1", "Q2"], var_name="half", value_name="revenue")
    print(f"\n  Melt of Q1/Q2 only (H1 comparison):")
    print(h1.to_string(index=False))


# ==============================================================
# 3. pivot(): Long to Wide
# ==============================================================
# pivot() is the inverse of melt. index= sets row labels, columns=
# sets new column names from a key column, and values= fills cells.
# Every index × column pair must be unique — use pivot_table()
# when duplicates exist and an aggregation function is needed.

def demo_pivot() -> None:
    wide = make_wide_df()
    long = wide.melt(
        id_vars=["rep", "region"], value_vars=["Q1", "Q2", "Q3", "Q4"],
        var_name="quarter", value_name="revenue",
    )

    # Pivot back to wide — verifies the melt round-trip
    restored = long.pivot(index=["rep", "region"], columns="quarter", values="revenue")
    restored.columns.name = None
    restored = restored.reset_index()
    print(f"\n  pivot() round-trip back to wide:")
    print(restored.to_string(index=False))

    # Cross-tab style: rep vs. quarter (no region column)
    rep_quarter = long.pivot(index="rep", columns="quarter", values="revenue")
    rep_quarter.columns.name = None
    print(f"\n  Revenue pivot: rep × quarter:")
    print(rep_quarter.to_string())

    # Confirm the values match the original dataset
    cols = ["rep", "Q1", "Q2", "Q3", "Q4"]
    match = (
        restored[cols].reset_index(drop=True)
        .equals(wide[cols].reset_index(drop=True))
    )
    print(f"\n  Round-trip matches original: {match}")


# ==============================================================
# 4. stack(): Columns into a Row MultiIndex
# ==============================================================
# stack() rotates the innermost column level into a new innermost
# row level, producing a Series (or DataFrame) with a MultiIndex.
# It is useful when all measurement columns share the same type
# and you want to process them uniformly via groupby or iteration.

def demo_stack() -> None:
    wide = make_wide_df().set_index("rep")

    # Stack Q1–Q4 into a (rep, quarter) MultiIndex Series
    stacked = wide[["Q1", "Q2", "Q3", "Q4"]].stack()
    stacked.index.names = ["rep", "quarter"]
    stacked.name = "revenue"
    print(f"\n  stack() result — Series with MultiIndex:")
    print(stacked.to_string())

    # Flatten to a regular DataFrame with reset_index
    flat = stacked.reset_index()
    print(f"\n  After reset_index():")
    print(flat.to_string(index=False))

    # Best quarter per rep (highest revenue)
    best = flat.loc[flat.groupby("rep")["revenue"].idxmax()][["rep", "quarter", "revenue"]]
    print(f"\n  Best quarter per rep:")
    print(best.to_string(index=False))


# ==============================================================
# 5. unstack(): Row Level into Columns
# ==============================================================
# unstack() is the inverse of stack. It rotates one level of the
# row MultiIndex out into column labels. The default is the innermost
# level, but a level name or integer can be passed. fill_value=
# replaces any NaN that arises when a combination is missing.

def demo_unstack() -> None:
    wide = make_wide_df().set_index("rep")
    stacked = wide[["Q1", "Q2", "Q3", "Q4"]].stack()
    stacked.index.names = ["rep", "quarter"]
    stacked.name = "revenue"

    # Unstack the quarter level back to columns
    unstacked = stacked.unstack("quarter")
    unstacked.columns.name = None
    print(f"\n  unstack('quarter') — back to wide:")
    print(unstacked.to_string())

    # Three-level index: region → rep → quarter
    long = make_wide_df().melt(
        id_vars=["rep", "region"], value_vars=["Q1", "Q2"],
        var_name="quarter", value_name="revenue",
    )
    mi = long.set_index(["region", "rep", "quarter"])["revenue"]
    print(f"\n  unstack('quarter') on 3-level MultiIndex:")
    print(mi.unstack("quarter").to_string())


# ==============================================================
# 6. crosstab(): Frequency Tables
# ==============================================================
# pd.crosstab() computes a contingency table between two categorical
# variables. normalize= converts counts to proportions (by 'index',
# 'columns', or 'all'). Pass values= and aggfunc= to aggregate a
# numeric column instead of just counting rows.

def demo_crosstab() -> None:
    np.random.seed(7)
    n = 40
    sales_log = pd.DataFrame({
        "region":  np.random.choice(["North", "South", "East", "West"], n),
        "quarter": np.random.choice(["Q1", "Q2", "Q3", "Q4"], n),
        "product": np.random.choice(["Widget", "Gadget"], n),
        "revenue": np.random.randint(5_000, 20_000, n),
    })

    # Deal count by region × quarter
    ct = pd.crosstab(sales_log["region"], sales_log["quarter"])
    print(f"\n  crosstab: deal count by region × quarter:")
    print(ct.to_string())

    # Proportions within each region (rows sum to 1)
    ct_pct = pd.crosstab(
        sales_log["region"], sales_log["quarter"], normalize="index"
    ).round(2)
    print(f"\n  Normalised by row (proportions within each region):")
    print(ct_pct.to_string())

    # Total revenue instead of counts
    ct_rev = pd.crosstab(
        sales_log["region"], sales_log["quarter"],
        values=sales_log["revenue"], aggfunc="sum",
    ).fillna(0).astype(int)
    print(f"\n  crosstab: total revenue by region × quarter:")
    print(ct_rev.to_string())


# ==============================================================
# 7. Practical Example: Survey Results Reshaping
# ==============================================================
# Reshape a wide survey export to long format, then compute per-
# question and per-respondent summaries. Survey tools typically
# deliver one column per question — melt is the right first step.

def demo_survey_reshape() -> None:
    np.random.seed(0)
    n = 8
    survey = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "department":    np.random.choice(["Engineering", "Marketing", "Sales"], n),
        "Q_clarity":     np.random.randint(1, 6, n),
        "Q_support":     np.random.randint(1, 6, n),
        "Q_workload":    np.random.randint(1, 6, n),
        "Q_growth":      np.random.randint(1, 6, n),
    })
    print(f"\n  --- Raw wide survey ({survey.shape[0]} respondents) ---")
    print(survey.to_string(index=False))

    # Melt to long: one row per respondent × question
    long = survey.melt(
        id_vars=["respondent_id", "department"],
        value_vars=["Q_clarity", "Q_support", "Q_workload", "Q_growth"],
        var_name="question",
        value_name="score",
    )
    long["question"] = long["question"].str.replace("Q_", "", regex=False)
    print(f"\n  --- Long format ({long.shape[0]} rows) ---")
    print(long.to_string(index=False))

    # Mean score per question (ascending — lower means harder)
    q_means = long.groupby("question")["score"].mean().round(2).sort_values()
    print(f"\n  Mean score per question (ascending):")
    print(q_means.to_string())

    # Department × question heatmap via unstack
    dept_q = (
        long.groupby(["department", "question"])["score"]
        .mean().round(2)
        .unstack("question")
    )
    dept_q.columns.name = None
    print(f"\n  Mean score by department × question:")
    print(dept_q.to_string())

    # Respondent overall average
    resp_avg = long.groupby("respondent_id")["score"].mean().round(2)
    print(f"\n  Average score per respondent:")
    print(resp_avg.to_string())
    print(f"\n  Lowest-scoring respondent : id={resp_avg.idxmin()}  avg={resp_avg.min():.2f}")
    print(f"  Highest-scoring respondent: id={resp_avg.idxmax()}  avg={resp_avg.max():.2f}")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Wide vs. Long Format")
    demo_wide_vs_long()

    section("2. melt(): Wide to Long")
    demo_melt()

    section("3. pivot(): Long to Wide")
    demo_pivot()

    section("4. stack(): Columns into a Row MultiIndex")
    demo_stack()

    section("5. unstack(): Row Level into Columns")
    demo_unstack()

    section("6. crosstab(): Frequency Tables")
    demo_crosstab()

    section("7. Practical Example: Survey Results Reshaping")
    demo_survey_reshape()


if __name__ == "__main__":
    main()

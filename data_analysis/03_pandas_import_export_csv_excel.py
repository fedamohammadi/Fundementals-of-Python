"""
Pandas Import and Export — CSV and Excel:
- Writing a CSV from scratch using the csv module
- Reading CSV with pandas: read_csv and key parameters
- Writing CSV with pandas: to_csv
- Reading and writing Excel: read_excel and to_excel
- Common read_csv parameters: sep, header, usecols, dtype, parse_dates
- Load-time filtering: skiprows, nrows, na_values
- Practical workflow: load → inspect → clean → export
"""

import csv
import io
import os
import tempfile
import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: employee records used across all sections
# ==============================================================

N_ROWS = 12


def make_employees() -> pd.DataFrame:
    rng    = np.random.default_rng(42)
    depts  = rng.choice(["Eng", "Sales", "HR", "Finance"], N_ROWS)
    years  = rng.integers(1, 15, N_ROWS).astype(float)
    salary = (40_000 + years * 3_500 + rng.normal(0, 5_000, N_ROWS)).round(0)
    rating = rng.choice(["good", "excellent", "needs_improvement"], N_ROWS)
    hired  = pd.date_range("2010-01-01", periods=N_ROWS, freq="3ME")
    return pd.DataFrame({
        "emp_id":    range(101, 101 + N_ROWS),
        "dept":      depts,
        "yrs_exp":   years,
        "salary":    salary,
        "rating":    rating,
        "hire_date": hired,
    })


# ==============================================================
# 1. Writing a CSV from Scratch (csv module)
# ==============================================================
# The stdlib csv module shows what a CSV file actually is: plain
# delimited text. Understanding the raw format prevents confusion
# when pandas read_csv behaves unexpectedly due to quoting or encoding.

def demo_write_csv_stdlib(path: str) -> None:
    rows = [
        ["id", "city",         "population"],
        [1,    "New York",     8_336_817],
        [2,    "Los Angeles",  3_979_576],
        [3,    "Chicago",      2_693_976],
        [4,    "Houston",      2_320_268],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n  Wrote CSV to: {path}")
    print(f"\n  Raw file content:")
    with open(path, encoding="utf-8") as f:
        for line in f:
            print(f"    {line}", end="")


# ==============================================================
# 2. Reading CSV with pandas
# ==============================================================
# pd.read_csv is the entry point for almost all tabular data.
# The defaults handle the common case; learning the key parameters
# saves hours debugging real-world files.

def demo_read_csv(path: str) -> None:
    # Basic read — pandas infers dtypes automatically
    df = pd.read_csv(path)
    print(f"\n  pd.read_csv result:")
    print(df.to_string(index=False))
    print(f"\n  dtypes:\n{df.dtypes}")
    print(f"\n  shape: {df.shape}")

    # Use a column as the index at read time
    df_idx = pd.read_csv(path, index_col="id")
    print(f"\n  index_col='id':")
    print(df_idx.to_string())


# ==============================================================
# 3. Writing CSV with pandas
# ==============================================================
# df.to_csv() is the inverse of read_csv. The most important
# parameter is index=False — by default pandas writes the index
# as an extra first column, which clutters the file on re-read.

def demo_to_csv(df: pd.DataFrame, base_path: str) -> None:
    # index=False: clean file with no unnamed column on re-read
    df.to_csv(base_path + "_no_index.csv", index=False)
    print(f"\n  Wrote (no index): {base_path}_no_index.csv")

    # Custom delimiter and column subset
    df[["emp_id", "dept", "salary"]].to_csv(
        base_path + "_pipe.csv", sep="|", index=False
    )
    print(f"  Wrote pipe-delimited subset: {base_path}_pipe.csv")

    # Round-trip verification
    df_back = pd.read_csv(base_path + "_no_index.csv")
    print(f"\n  Round-trip shape matches: {df.shape == df_back.shape}")
    print(f"\n  First 3 rows of round-tripped file:")
    print(df_back.head(3).to_string(index=False))


# ==============================================================
# 4. Reading and Writing Excel
# ==============================================================
# read_excel and to_excel mirror their CSV counterparts but also
# handle multiple sheets. Requires openpyxl for .xlsx files.

def demo_excel(df: pd.DataFrame, base_path: str) -> None:
    xlsx_path = base_path + ".xlsx"

    # Write multiple sheets in one pass using ExcelWriter
    summary = (
        df.groupby("dept")["salary"]
        .agg(count="count", mean_salary="mean")
        .round(0)
        .reset_index()
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="employees", index=False)
        summary.to_excel(writer, sheet_name="dept_summary", index=False)

    print(f"\n  Wrote Excel workbook: {xlsx_path}")
    print(f"  Sheets written: employees, dept_summary")

    # Read back a specific sheet
    df_emp = pd.read_excel(xlsx_path, sheet_name="employees")
    print(f"\n  'employees' sheet (first 3 rows):")
    print(df_emp.head(3).to_string(index=False))

    # Read back the summary sheet
    df_summ = pd.read_excel(xlsx_path, sheet_name="dept_summary")
    print(f"\n  'dept_summary' sheet:\n{df_summ.to_string(index=False)}")


# ==============================================================
# 5. Common read_csv Parameters
# ==============================================================
# Real-world CSVs rarely match the happy-path defaults. These
# parameters handle the most frequent deviations encountered in
# practice: non-comma delimiters, missing headers, column subsets,
# wrong dtypes, and date columns stored as strings.

def demo_read_csv_params() -> None:
    # sep: tab-delimited files
    tsv = "name\tage\tcity\nAlice\t30\tNY\nBob\t25\tLA\n"
    df_tsv = pd.read_csv(io.StringIO(tsv), sep="\t")
    print(f"\n  sep='\\t' (tab-delimited):\n{df_tsv.to_string(index=False)}")

    # header=None: file has no header row — supply names manually
    no_hdr = "Alice,30,NY\nBob,25,LA\n"
    df_nh = pd.read_csv(
        io.StringIO(no_hdr),
        header=None,
        names=["name", "age", "city"],
    )
    print(f"\n  header=None with names=:\n{df_nh.to_string(index=False)}")

    # usecols: read only the columns you need — saves memory on large files
    full = "id,name,age,income,score\n1,Alice,30,52000,88\n2,Bob,25,47000,72\n"
    df_cols = pd.read_csv(io.StringIO(full), usecols=["id", "name", "income"])
    print(f"\n  usecols=['id','name','income']:\n{df_cols.to_string(index=False)}")

    # dtype: prevent leading-zero loss on ID-like fields
    id_csv = "emp_id,value\n001,100\n002,200\n003,300\n"
    df_id = pd.read_csv(io.StringIO(id_csv), dtype={"emp_id": str})
    print(f"\n  dtype={{'emp_id': str}} — preserves leading zeros:")
    print(df_id.to_string(index=False))
    print(f"  emp_id dtype: {df_id['emp_id'].dtype}")

    # parse_dates: read a column as datetime instead of plain string
    date_csv = "date,value\n2024-01-01,100\n2024-02-01,110\n2024-03-01,105\n"
    df_dates = pd.read_csv(io.StringIO(date_csv), parse_dates=["date"])
    print(f"\n  parse_dates=['date']:\n{df_dates.to_string(index=False)}")
    print(f"  date dtype: {df_dates['date'].dtype}")


# ==============================================================
# 6. Load-time Filtering: skiprows, nrows, na_values
# ==============================================================
# These parameters trim or adjust data at read time — before the
# DataFrame is constructed — keeping both the intent and the code clear.

def demo_load_filtering() -> None:
    # skiprows: skip metadata comment lines at the top
    messy = (
        "# Source: internal HR survey, 2024\n"
        "# Units: USD\n"
        "name,salary\n"
        "Alice,52000\n"
        "Bob,47000\n"
    )
    df_skip = pd.read_csv(io.StringIO(messy), skiprows=2)
    print(f"\n  skiprows=2 (skip 2 comment lines):\n{df_skip.to_string(index=False)}")

    # nrows: preview the first n rows of a huge file without reading it all
    big = "id,x\n" + "".join(f"{i},{i * 2}\n" for i in range(1, 101))
    df_head = pd.read_csv(io.StringIO(big), nrows=5)
    print(f"\n  nrows=5 (preview of 100-row file):\n{df_head.to_string(index=False)}")

    # na_values: treat custom sentinel strings as NaN
    na_csv = "name,score\nAlice,88\nBob,N/A\nClara,-99\nDavid,75\n"
    df_na = pd.read_csv(io.StringIO(na_csv), na_values=["N/A", -99])
    print(f"\n  na_values=['N/A', -99]:\n{df_na.to_string(index=False)}")
    print(f"  NaN count in score: {df_na['score'].isna().sum()}")

    # Combined: all three parameters together
    combined = (
        "# Metadata\n"
        "name,age,country\n"
        "Alice,30,USA\n"
        "Bob,UNKNOWN,GBR\n"
        "Clara,28,DEU\n"
        "David,35,USA\n"
    )
    df_all = pd.read_csv(
        io.StringIO(combined),
        skiprows=1,
        nrows=3,
        na_values=["UNKNOWN"],
    )
    print(f"\n  Combined (skiprows=1, nrows=3, na_values=['UNKNOWN']):")
    print(df_all.to_string(index=False))

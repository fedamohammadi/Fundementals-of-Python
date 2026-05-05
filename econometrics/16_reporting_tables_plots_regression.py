"""
Reporting Tables, Plots, and Regression Output:
- Summary statistics tables: formatted for applied econometrics
- Regression output tables: comparing multiple models side by side
- Coefficient plots: visualizing estimates and confidence intervals
- Residual diagnostic plots (text-based, matplotlib-free)
- Model comparison tables: AIC, BIC, RMSE across specifications
- Exporting results: CSV and LaTeX formats
- A practical guide for reproducible reporting
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as ts_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Cross-sectional dataset: wage regression.
#   True model: log(wage) = 0.4 + 0.08*educ + 0.05*exper + 0.3*female + eps
#
# Time series dataset: AR(1) for residual diagnostic demonstrations.

N_CS = 500
N_TS = 200


def make_wages(seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    educ  = rng.integers(8, 21, N_CS).astype(float)
    exper = rng.integers(0, 31, N_CS).astype(float)
    female = rng.binomial(1, 0.45, N_CS).astype(float)
    log_wage = 0.4 + 0.08*educ + 0.05*exper + 0.3*female + rng.normal(0, 0.4, N_CS)
    return pd.DataFrame({
        "log_wage": log_wage,
        "educ":     educ,
        "exper":    exper,
        "exper2":   exper ** 2 / 100,
        "female":   female,
    })


def make_ts_data(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, eps = np.zeros(N_TS), rng.normal(0, 1, N_TS)
    for t in range(1, N_TS):
        y[t] = 0.7 * y[t - 1] + eps[t]
    return y


DF_WAGES = make_wages()
Y_TS     = make_ts_data()


# ==============================================================
# 1. Summary Statistics Tables
# ==============================================================
# Summary statistics are the first table in any applied paper.
# Report: N, mean, std, min, median, max.
# Good practices:
#   - Round to 3 significant digits; avoid false precision.
#   - Note units or scale in variable names.
#   - List binary variables separately with their fraction of ones.
#   - Always report N prominently; partial samples must be noted.

def demo_summary_stats() -> None:
    df   = DF_WAGES
    cont = ["log_wage", "educ", "exper"]
    biny = ["female"]

    print(f"\n  Summary statistics  (N = {len(df)})")
    print()
    print(f"  {'Variable':>12} | {'Mean':>7} | {'Std':>7} | {'Min':>7} | {'Median':>7} | {'Max':>7}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for var in cont:
        col = df[var]
        print(f"  {var:>12} | {col.mean():>7.3f} | {col.std():>7.3f} | "
              f"{col.min():>7.3f} | {col.median():>7.3f} | {col.max():>7.3f}")

    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    print(f"  {'Variable':>12} | {'Fraction':>7} | (Binary 0/1)")
    for var in biny:
        print(f"  {var:>12} | {df[var].mean():>7.3f}")

    print()
    print("  log_wage: log of hourly wage in dollars.")
    print(f"  female: fraction female = {df['female'].mean():.2f}.")
    print("  Quick one-liner: df.describe().T  (includes quartiles).")

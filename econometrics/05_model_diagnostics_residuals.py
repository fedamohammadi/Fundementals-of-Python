"""
Model Diagnostics and Residual Analysis:
- What residuals are and why they carry diagnostic information
- Residual-vs-fitted plots: spotting patterns that violate OLS assumptions
- Heteroskedasticity: detecting and testing for non-constant variance
- Normality of residuals: Q-Q plots and the Jarque-Bera test
- Influential observations: leverage, studentized residuals, Cook's distance
- The Ramsey RESET test: detecting functional form misspecification
- A diagnostic checklist and what to do when checks fail

After fitting any OLS model, residuals are the first thing to examine.
A 'good-looking' R² is no substitute for verifying the assumptions.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as diag
import statsmodels.stats.stattools as stattools
from scipy import stats


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# We deliberately generate TWO datasets:
#
#   clean_data : satisfies all OLS assumptions
#       ln(wage) = 1.6 + 0.10*educ + 0.008*exper + eps
#       eps ~ N(0, 0.25²)  — homoskedastic errors
#
#   dirty_data : intentionally breaks assumptions so diagnostics
#       can detect the problems:
#       (a) heteroskedastic errors — variance grows with educ
#       (b) a misspecified functional form (left as level-level
#           when the true DGP is log-level)
#
# Comparing diagnostics across both datasets shows exactly what
# each test is sensitive to.

TRUE_LOG_INTERCEPT = 1.6
TRUE_LOG_EDUC      = 0.10    # 10% wage increase per year of schooling
TRUE_LOG_EXPER     = 0.008   # 0.8% wage increase per year of experience
CLEAN_ERROR_STD    = 0.25    # constant (homoskedastic) noise on log scale


def make_clean_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate data that satisfies all classical OLS assumptions.
    The DGP is log-level; we fit the correctly-specified log-level model.
    """
    rng = np.random.default_rng(seed)

    educ  = rng.integers(8, 22, size=n).astype(float)
    exper = rng.integers(0, 41, size=n).astype(float)
    eps   = rng.normal(0, CLEAN_ERROR_STD, size=n)

    log_wage = TRUE_LOG_INTERCEPT + TRUE_LOG_EDUC * educ + TRUE_LOG_EXPER * exper + eps

    return pd.DataFrame({
        "wage":     np.exp(log_wage),
        "log_wage": log_wage,
        "educ":     educ,
        "exper":    exper,
    })


def make_dirty_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate data with heteroskedastic errors: the noise variance
    increases linearly with years of education.

    This violates the OLS assumption of homoskedasticity and will
    produce a characteristic 'funnel' shape in residual plots.
    """
    rng = np.random.default_rng(seed)

    educ  = rng.integers(8, 22, size=n).astype(float)
    exper = rng.integers(0, 41, size=n).astype(float)

    # Error std grows with educ: more-educated workers have more wage dispersion
    error_std = 0.05 * educ          # std = 0.40 at educ=8, 1.10 at educ=22
    eps       = rng.normal(0, error_std, size=n)

    log_wage = TRUE_LOG_INTERCEPT + TRUE_LOG_EDUC * educ + TRUE_LOG_EXPER * exper + eps

    return pd.DataFrame({
        "wage":     np.exp(log_wage),
        "log_wage": log_wage,
        "educ":     educ,
        "exper":    exper,
    })


# ==============================================================
# 1. What Are Residuals?
# ==============================================================
# The OLS residual for observation i is:
#   e_i = y_i - ŷ_i   where ŷ_i = b0 + b1*x1_i + b2*x2_i + ...
#
# Residuals are our best estimate of the unobserved errors eps_i.
# The key OLS assumptions about eps translate into expected properties
# of residuals if the model is correctly specified:
#
#   Assumption                  | What residuals should look like
#   ----------------------------+----------------------------------
#   E[eps | X] = 0              | Mean residual ≈ 0 at every X value
#   Var[eps | X] = σ² (const)  | Spread of residuals constant across ŷ
#   eps ~ Normal               | Residuals fall near a straight Q-Q line
#   No influential outliers    | No single point drives the fit
#
# Residual plots let us eyeball all four assumptions at once.
# Formal tests quantify deviations from each assumption.

def demo_what_are_residuals() -> None:
    """
    Fit the correctly-specified log-level model on clean data,
    print the first few observed / fitted / residual values,
    and verify the two algebraic identities that always hold for OLS:
      (1) sum of residuals = 0
      (2) residuals are uncorrelated with fitted values
    """
    df     = make_clean_data()
    result = smf.ols("log_wage ~ educ + exper", data=df).fit()

    fitted    = result.fittedvalues
    residuals = result.resid

    print("\n  First 6 observations:")
    print(f"  {'i':>4} | {'log_wage (y)':>14} | {'fitted (ŷ)':>12} | {'residual (e)':>14}")
    print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*12}-+-{'-'*14}")
    for i in range(6):
        y_i  = df["log_wage"].iloc[i]
        yhat = fitted.iloc[i]
        e_i  = residuals.iloc[i]
        print(f"  {i+1:>4} | {y_i:>14.6f} | {yhat:>12.6f} | {e_i:>14.6f}")

    print()
    sum_resid   = residuals.sum()
    corr_ey_hat = np.corrcoef(residuals, fitted)[0, 1]
    print(f"  OLS identities (always true by construction):")
    print(f"    Sum of residuals       = {sum_resid:.2e}   (should be ~0)")
    print(f"    Corr(residuals, ŷ)     = {corr_ey_hat:.2e}  (should be ~0)")
    print()
    print(f"  Residual summary statistics:")
    print(f"    Mean  : {residuals.mean():.6f}")
    print(f"    Std   : {residuals.std():.6f}   (estimates σ ≈ {CLEAN_ERROR_STD})")
    print(f"    Min   : {residuals.min():.6f}")
    print(f"    Max   : {residuals.max():.6f}")
    print()
    print("  The OLS residual std is close to the true error std (0.25),")
    print("  which confirms the model is correctly specified on clean data.")


# ==============================================================
# 2. Residual-vs-Fitted Plot (Numerical Summary)
# ==============================================================
# The residual-vs-fitted plot (e_i on y-axis, ŷ_i on x-axis) is the
# single most important diagnostic plot.  Two patterns to look for:
#
#   Pattern 1 — Curvature (non-linearity):
#     Residuals form a U-shape or inverted-U around zero.
#     Cause: you fit a linear model to a non-linear relationship.
#     Fix:   add polynomial terms or use logs.
#
#   Pattern 2 — Funnel / fan shape (heteroskedasticity):
#     The spread of residuals grows (or shrinks) with ŷ.
#     Cause: the error variance is not constant.
#     Fix:   use robust standard errors, or WLS, or log-transform y.
#
# A clean model produces residuals scattered randomly around zero
# with constant width — no patterns, no trends.
#
# Since we cannot show actual plots in a terminal, we summarise
# the residual spread in bins of fitted values, which gives the
# same diagnostic information numerically.

def residual_spread_by_bin(fitted: pd.Series, residuals: pd.Series,
                           n_bins: int = 5) -> None:
    """
    Divide the range of fitted values into n_bins equal-width bins.
    Print the mean and std of residuals in each bin.

    In a homoskedastic model the std should be roughly constant
    across bins.  A growing std signals heteroskedasticity.
    """
    bins      = pd.cut(fitted, bins=n_bins)
    bin_stats = residuals.groupby(bins, observed=True).agg(
        count="count",
        mean="mean",
        std="std",
    )

    print(f"\n  {'Fitted value bin':>32} | {'N':>5} | {'Mean resid':>12} | {'Std resid':>10}")
    print(f"  {'-'*32}-+-{'-'*5}-+-{'-'*12}-+-{'-'*10}")
    for bin_label, row in bin_stats.iterrows():
        print(f"  {str(bin_label):>32} | {int(row['count']):>5} | "
              f"{row['mean']:>12.4f} | {row['std']:>10.4f}")


def demo_residual_vs_fitted() -> None:
    """
    Compare the residual spread across fitted-value bins for:
      (a) correctly-specified model on clean (homoskedastic) data
      (b) same model specification on heteroskedastic data
    """
    df_clean = make_clean_data()
    df_dirty = make_dirty_data()

    fit_clean = smf.ols("log_wage ~ educ + exper", data=df_clean).fit()
    fit_dirty = smf.ols("log_wage ~ educ + exper", data=df_dirty).fit()

    print("\n  --- Clean data (homoskedastic errors) ---")
    print("  Expectation: residual std roughly the same across all bins.")
    residual_spread_by_bin(fit_clean.fittedvalues, fit_clean.resid)

    print("\n  --- Dirty data (heteroskedastic errors: std grows with educ) ---")
    print("  Expectation: residual std increases in higher fitted-value bins.")
    residual_spread_by_bin(fit_dirty.fittedvalues, fit_dirty.resid)

    print()
    std_clean = fit_clean.resid.std()
    std_dirty = fit_dirty.resid.std()
    print(f"  Overall residual std — clean: {std_clean:.4f}  |  dirty: {std_dirty:.4f}")
    print("  The dirty dataset has larger overall variance because higher-educ")
    print("  observations contribute disproportionately large residuals.")

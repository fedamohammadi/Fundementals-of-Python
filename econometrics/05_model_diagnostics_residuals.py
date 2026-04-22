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


# ==============================================================
# 3. Testing for Heteroskedasticity
# ==============================================================
# Heteroskedasticity means Var[eps_i | X] is not constant.
# OLS estimates remain unbiased, but standard errors are WRONG —
# which means t-tests, p-values, and confidence intervals are invalid.
#
# Two standard formal tests:
#
#   Breusch-Pagan (BP) test
#     H0: error variance is constant (homoskedastic)
#     Procedure: regress squared residuals on the regressors.
#       If regressors explain the squared residuals, variance is
#       not constant.  The test statistic is LM = n * R² from that
#       auxiliary regression; it follows χ²(k) under H0.
#     Best when: heteroskedasticity is a linear function of X.
#
#   White test
#     H0: same as BP
#     Procedure: like BP, but the auxiliary regression also includes
#       squares and cross-products of all regressors.
#     Best when: heteroskedasticity has a more general (non-linear) form.
#     Limitation: uses many degrees of freedom in large models.
#
# If H0 is rejected:
#   - Report heteroskedasticity-robust (HC) standard errors (file 06).
#   - Or transform the model (e.g., log y) to reduce the variance spread.
#   - Or use weighted least squares (WLS) if you know the variance function.

def run_het_tests(result, label: str) -> None:
    """
    Run the Breusch-Pagan and White tests on a fitted OLS result.
    Print the test statistic, p-value, and a plain-English verdict.
    """
    # Breusch-Pagan test
    bp_stat, bp_pval, _, _ = diag.het_breuschpagan(result.resid, result.model.exog)

    # White test
    wh_stat, wh_pval, _, _ = diag.het_white(result.resid, result.model.exog)

    print(f"\n  [{label}]")
    print(f"  {'Test':>20} | {'Statistic':>10} | {'p-value':>10} | Verdict (α=0.05)")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*25}")
    for name, stat, pval in [("Breusch-Pagan", bp_stat, bp_pval),
                              ("White",         wh_stat, wh_pval)]:
        verdict = "REJECT H0 (heterosked.)" if pval < 0.05 else "Fail to reject H0"
        print(f"  {name:>20} | {stat:>10.4f} | {pval:>10.4f} | {verdict}")


def demo_heteroskedasticity_tests() -> None:
    """
    Run BP and White tests on both the clean and dirty datasets.
    The clean model should pass (fail to reject H0).
    The dirty model should fail (reject H0), detecting the planted
    heteroskedasticity.
    """
    df_clean = make_clean_data()
    df_dirty = make_dirty_data()

    fit_clean = smf.ols("log_wage ~ educ + exper", data=df_clean).fit()
    fit_dirty = smf.ols("log_wage ~ educ + exper", data=df_dirty).fit()

    run_het_tests(fit_clean, "Clean data — expect H0 not rejected")
    run_het_tests(fit_dirty, "Dirty data — expect H0 rejected")

    print()
    print("  Interpretation:")
    print("    Clean: both tests fail to reject H0 — no evidence of heteroskedasticity.")
    print("    Dirty: both tests reject H0 — variance is not constant across observations.")
    print()
    print("  What to do when you detect heteroskedasticity:")
    print("    1. Use robust standard errors (HC1 or HC3) — see file 06.")
    print("    2. Check whether a log transformation of y removes the pattern.")
    print("    3. If the variance function is known, use weighted least squares (WLS).")


# ==============================================================
# 4. Normality of Residuals
# ==============================================================
# OLS is unbiased and efficient regardless of whether eps is Normal.
# Normality matters for:
#   - Exact t and F distributions in small samples.
#   - Prediction intervals (which assume Normal errors).
#   - Maximum-likelihood interpretation of OLS.
#
# In large samples (n > ~100) the Central Limit Theorem makes
# the t and F distributions approximately correct even without
# normality, so this assumption is much less critical than
# homoskedasticity.
#
# Two common diagnostics:
#
#   Q-Q plot (quantile-quantile plot)
#     Plot sorted residual quantiles against theoretical Normal quantiles.
#     If residuals are Normal, the points fall on a straight line.
#     Heavy tails bow outward; skewness tilts the line.
#     (We summarise this numerically since we cannot draw a plot.)
#
#   Jarque-Bera test
#     H0: residuals are Normally distributed.
#     Test statistic combines skewness and excess kurtosis:
#       JB = (n/6) * [S² + (K-3)²/4]
#       where S = skewness, K = kurtosis.
#     JB ~ χ²(2) under H0.
#     Sensitive to large n — even tiny non-normality can be detected
#     in samples of 500+, even when it does not matter in practice.

def qq_summary(residuals: pd.Series, label: str) -> None:
    """
    Numerically summarise the Q-Q comparison by checking whether
    the 5th, 25th, 75th, and 95th percentiles of the residuals
    match the theoretical Normal percentiles at the same probabilities.

    Large gaps indicate heavy tails or skewness.
    """
    probs = [0.05, 0.25, 0.75, 0.95]
    r_std = (residuals - residuals.mean()) / residuals.std()   # standardize

    print(f"\n  [{label}]")
    print(f"  {'Percentile':>12} | {'Residual quantile':>18} | {'Normal quantile':>16} | {'Gap':>8}")
    print(f"  {'-'*12}-+-{'-'*18}-+-{'-'*16}-+-{'-'*8}")
    for p in probs:
        r_q = float(np.quantile(r_std, p))
        n_q = float(stats.norm.ppf(p))
        gap = r_q - n_q
        print(f"  {p*100:>11.0f}% | {r_q:>18.4f} | {n_q:>16.4f} | {gap:>8.4f}")


def demo_normality_tests() -> None:
    """
    Test residual normality on clean and dirty data using:
      (a) a numerical Q-Q summary (percentile comparison)
      (b) the Jarque-Bera test
    Also print skewness and kurtosis for direct interpretation.
    """
    df_clean = make_clean_data()
    df_dirty = make_dirty_data()

    fit_clean = smf.ols("log_wage ~ educ + exper", data=df_clean).fit()
    fit_dirty = smf.ols("log_wage ~ educ + exper", data=df_dirty).fit()

    for fit, label in [(fit_clean, "Clean (homosked.) data"),
                       (fit_dirty, "Dirty (heterosked.) data")]:
        resid = fit.resid

        # Skewness and kurtosis
        skew = float(stats.skew(resid))
        kurt = float(stats.kurtosis(resid))   # excess kurtosis (Normal = 0)

        # Jarque-Bera test
        jb_stat, jb_pval = stattools.jarque_bera(resid)[:2]

        print(f"\n  [{label}]")
        print(f"    Skewness       : {skew:>8.4f}  (Normal = 0)")
        print(f"    Excess kurtosis: {kurt:>8.4f}  (Normal = 0; >0 = heavy tails)")
        print(f"    Jarque-Bera    : stat={jb_stat:.4f},  p={jb_pval:.4f}  "
              f"({'reject H0' if jb_pval < 0.05 else 'fail to reject H0'} at α=0.05)")

        qq_summary(resid, "Q-Q percentile check")

    print()
    print("  Key takeaway:")
    print("    In large samples (n=500 here), the JB test is very sensitive.")
    print("    Even mild non-normality in the dirty data triggers rejection.")
    print("    This matters less for inference than heteroskedasticity does,")
    print("    but matters for prediction intervals and likelihood-based tests.")

"""
Cointegration and Error Correction Models:
- Cointegration: when two I(1) series share a long-run equilibrium
- The Engle-Granger two-step procedure
- Johansen trace and max-eigenvalue tests
- The error correction model (ECM): short-run dynamics + long-run adjustment
- Estimating the ECM with statsmodels VECM
- Interpreting cointegration and ECM output
- A practical guide for applied cointegration analysis
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Three simulated scenarios:
#
#   Cointegrated pair (x, y): share a common stochastic trend.
#     x_t = x_{t-1} + eps_x          (random walk)
#     y_t = BETA * x_t + mu_t        (long-run: y = BETA * x)
#     mu_t = 0.8*mu_{t-1} + eps_mu   (stationary deviations from equilibrium)
#   -> y - BETA*x is I(0); true BETA = 2.0
#
#   Independent random walks: unrelated I(1) series (spurious regression).
#
#   Trivariate system: three series driven by one common stochastic trend
#   (cointegrating rank = 1).

N          = 300
BETA       = 2.0
ALPHA_ECM  = -0.3    # true speed of adjustment


def make_cointegrated(n: int = N, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    x     = np.cumsum(rng.normal(0, 1, n))
    mu    = np.zeros(n)
    for t in range(1, n):
        mu[t] = 0.8 * mu[t - 1] + rng.normal(0, 0.5)
    y = BETA * x + mu
    return pd.DataFrame({"x": x, "y": y, "mu": mu})


def make_independent_rw(n: int = N, seed: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": np.cumsum(rng.normal(0, 1, n)),
        "y": np.cumsum(rng.normal(0, 1, n)),
    })


def make_vecm_data(n: int = N, seed: int = 5) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "x1": trend + rng.normal(0, 0.5, n),
        "x2": 1.5 * trend + rng.normal(0, 0.5, n),
        "x3": -0.5 * trend + rng.normal(0, 0.5, n),
    })


# ==============================================================
# 1. The Concept of Cointegration
# ==============================================================
# Two I(1) series x_t and y_t are cointegrated if there exists beta such that:
#   z_t = y_t - beta * x_t  is I(0)  (stationary equilibrium error)
#
# The vector (1, -beta) is the cointegrating vector.
#
# Why it matters:
#   - OLS on levels of cointegrated series is "super-consistent" (faster
#     convergence than standard sqrt(N)) but standard errors are NOT valid.
#   - Granger Representation Theorem: cointegrated series have an ECM
#     representation capturing both short- and long-run dynamics.
#   - Differencing throws away the long-run relationship -- do not difference
#     cointegrated series before modeling.
#
# Classic examples: consumption and income, money demand (M, P, Y), PPP.

def demo_cointegration_concept() -> None:
    df_coint = make_cointegrated()
    df_indep  = make_independent_rw()

    print(f"\n  Is y - {BETA}*x stationary?  (N = {N})")
    print()

    for label, df in [("Cointegrated pair", df_coint), ("Independent RWs", df_indep)]:
        spread = df["y"] - BETA * df["x"]
        adf_p  = adfuller(spread, regression="c", autolag="AIC")[1]
        corr   = df["x"].corr(df["y"])
        verdict = "cointegrated" if adf_p < 0.05 else "NOT cointegrated"
        print(f"  {label}:")
        print(f"    Corr(x, y) = {corr:.3f}   ADF-p(y - {BETA}x) = {adf_p:.4f}  -->  {verdict}")
        print()

    print("  Both pairs can show high correlation -- that is not sufficient.")
    print("  Only the cointegrated pair has a stationary spread (true long-run link).")
    print("  High R² between two I(1) series does NOT imply cointegration.")


# ==============================================================
# 2. Engle-Granger Two-Step Procedure
# ==============================================================
# The Engle-Granger (1987) test for cointegration:
#
# Step 1: Estimate the long-run relationship by OLS:
#   y_t = alpha + beta * x_t + z_hat_t
#
# Step 2: Test whether z_hat_t is I(0) using ADF.
#   H0: no cointegration (z_hat has a unit root).
#   statsmodels.tsa.stattools.coint() wraps this test.
#   Critical values are more negative than standard ADF (estimated residuals
#   lose one degree of freedom per variable in the cointegrating regression).
#
# Limitations:
#   - Only identifies one cointegrating vector.
#   - Results can depend on which variable is the LHS.
#   - For three or more variables, use the Johansen test.

def demo_engle_granger() -> None:
    df_coint = make_cointegrated()
    df_indep  = make_independent_rw()

    print(f"\n  Engle-Granger two-step cointegration test")
    print()
    print(f"  {'Dataset':>20} | {'OLS beta':>9} | {'EG stat':>9} | {'p-value':>9} | Conclusion")
    print(f"  {'-'*20}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*15}")

    for label, df in [("Cointegrated", df_coint), ("Independent RWs", df_indep)]:
        res      = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
        beta_hat = res.params["x"]
        eg_stat, eg_pval, _ = coint(df["y"], df["x"])
        verdict  = "cointegrated" if eg_pval < 0.05 else "NOT cointegrated"
        print(f"  {label:>20} | {beta_hat:>9.4f} | {eg_stat:>9.4f} | {eg_pval:>9.4f} | {verdict}")

    print()
    print(f"  True beta = {BETA:.1f}")
    df = make_cointegrated()
    res = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    print(f"  OLS beta_hat = {res.params['x']:.4f}  (super-consistent; faster than 1/sqrt(N))")
    print(f"  OLS t-stat p = {res.pvalues['x']:.4f}  WARNING: invalid -- use EG test instead")
    print()
    print("  EG critical values are more negative than standard ADF because")
    print("  residuals are estimated (not observed), shifting the distribution.")


# ==============================================================
# 3. Johansen Test
# ==============================================================
# The Johansen test handles multivariate systems and identifies the
# number of cointegrating vectors (the cointegrating rank r).
#
# The VECM representation:
#   ΔY_t = Π Y_{t-1} + Γ_1 ΔY_{t-1} + ... + Γ_{k-1} ΔY_{t-k+1} + eps_t
#   Π = αβ'  where β = cointegrating vectors, α = adjustment coefficients.
#
# Two test statistics:
#   Trace:   H0: rank(Π) ≤ r  vs H1: rank > r   (tests jointly)
#   Max-λ:   H0: rank(Π) = r  vs H1: rank = r+1  (tests one at a time)
#
# Procedure: start with r=0, increase until H0 is not rejected.
#   r=0: no long-run relation -> use VAR in first differences.
#   0 < r < k: use VECM with that rank.
#   r=k: all series are I(0) -> no cointegration needed.

def demo_johansen_test() -> None:
    df_coint = make_cointegrated()[["x", "y"]]
    df_vec   = make_vecm_data()

    for label, df, true_rank in [
        ("Bivariate cointegrated (true rank=1)", df_coint, 1),
        ("Trivariate, one common trend (true rank=1)", df_vec, 1),
    ]:
        print(f"\n  Johansen trace test: {label}")
        print(f"  (k_ar_diff=2, deterministic='ci')")
        print()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = select_coint_rank(df.values, det_order=0, k_ar_diff=2,
                                       method="trace", signif=0.05)
        print(f"  Selected rank = {result.rank}  (true rank = {true_rank})")
        print()
        print(f"  {'H0':>10} | {'Trace stat':>12} | {'5% CV':>8} | Decision")
        print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}")
        for i, (stat, cv) in enumerate(zip(result.test_stats, result.crit_vals)):
            decision = "reject H0" if stat > cv else "fail to reject"
            print(f"  {'rank ≤ '+str(i):>10} | {stat:>12.4f} | {cv:>8.4f} | {decision}")

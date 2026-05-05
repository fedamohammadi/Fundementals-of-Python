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

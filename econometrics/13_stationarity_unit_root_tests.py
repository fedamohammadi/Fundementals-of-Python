"""
Stationarity and Unit Root Tests:
- Unit roots and integration orders: I(0), I(1), I(2)
- The Augmented Dickey-Fuller (ADF) test: three specifications
- The KPSS test: testing stationarity as the null hypothesis
- The Phillips-Perron test: nonparametric correction for autocorrelation
- Deterministic vs. stochastic trends
- Seasonal integration and testing
- A practical decision framework for applied work
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Four canonical processes illustrating integration orders:
#
#   I(0): stationary AR(1) with phi = 0.6
#   I(1): random walk (unit root)
#   I(2): double unit root (cumsum of random walk)
#   Trend-stationary: y_t = 0.5 + 0.1*t + eps  (I(0) around a linear trend)
#
# Seasonal: quarterly series with stochastic trend + fixed seasonal pattern.

N = 250


def make_i0(phi: float = 0.6, n: int = N, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, eps = np.zeros(n), rng.normal(0, 1, n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def make_i1(n: int = N, seed: int = 42) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0, 1, n))


def make_i2(n: int = N, seed: int = 42) -> np.ndarray:
    return np.cumsum(np.cumsum(np.random.default_rng(seed).normal(0, 1, n)))


def make_trend_stationary(n: int = N, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = np.arange(n)
    return 0.5 + 0.1 * t + rng.normal(0, 2, n)


def make_seasonal_i1(n: int = N, seed: int = 42) -> np.ndarray:
    rng     = np.random.default_rng(seed)
    seasons = np.array([0.0, 1.5, 0.5, -2.0])
    trend   = np.cumsum(rng.normal(0, 0.5, n))
    seas    = np.array([seasons[t % 4] for t in range(n)])
    return trend + seas + rng.normal(0, 0.3, n)


# ==============================================================
# 1. Unit Roots and Integration Orders
# ==============================================================
# A series y_t has a unit root if:
#   y_t = y_{t-1} + eps_t  (random walk; Var(y_t) = t*sigma² grows)
#
# Integration order I(d):
#   I(0): stationary in levels.
#   I(1): first difference is I(0); levels have a unit root.
#   I(2): second difference is I(0); first difference has a unit root.
#
# Most macroeconomic series (GDP, prices, rates) are I(1).
# Returns and growth rates are typically I(0).
#
# Spurious regression: OLS on two unrelated I(1) series gives high R²
# and significant t-stats even with no true relationship.

def demo_integration_orders() -> None:
    series = {
        "I(0) AR(1) phi=0.6":    make_i0(),
        "I(1) random walk":      make_i1(),
        "I(2) double unit root": make_i2(),
        "Trend-stationary":      make_trend_stationary(),
    }

    print(f"\n  Integration order diagnostics  (N = {N})")
    print()
    print(f"  {'Series':>25} | {'Mean':>8} | {'Std':>8} | {'Diff Std':>9} | Classification")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*20}")

    for label, y in series.items():
        diff_std = np.diff(y).std()
        if "I(0)" in label:
            cls = "stationary"
        elif "I(1)" in label:
            cls = "unit root"
        elif "I(2)" in label:
            cls = "unit root × 2"
        else:
            cls = "trend-stationary"
        print(f"  {label:>25} | {y.mean():>8.3f} | {y.std():>8.3f} | {diff_std:>9.3f} | {cls}")

    print()
    print("  I(1): Var(y_t) grows; differencing produces I(0).")
    print("  Trend-stationary: detrend (not difference) to achieve stationarity.")
    print("  Over-differencing an I(0) series introduces an MA unit root -- avoid it.")


# ==============================================================
# 2. Augmented Dickey-Fuller Test
# ==============================================================
# ADF tests H0: unit root vs. H1: stationary.
# Augments the basic DF regression with lagged differences:
#
#   Δy_t = α + β*t + ρ*y_{t-1} + Σ γ_j*Δy_{t-j} + eps_t
#
# Three regression specifications:
#   'n': no constant, no trend    (demeaned, near-zero series)
#   'c': constant only            (most common; series without trend)
#   'ct': constant + trend        (series with a visible deterministic trend)
#
# Decision: reject H0 (unit root) if ADF stat < critical value or p < 0.05.
# Lag selection: 'AIC' selects augmenting lags automatically.

def demo_adf_test() -> None:
    series = {
        "I(0) AR(1)":       make_i0(),
        "I(1) random walk": make_i1(),
        "Trend-stationary": make_trend_stationary(),
    }
    specs = [("c", "constant only"), ("ct", "const + trend")]

    for label, y in series.items():
        print(f"\n  ADF test: {label}")
        print(f"  {'Specification':>20} | {'ADF stat':>10} | {'p-value':>9} | {'Lags':>5} | Verdict")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*9}-+-{'-'*5}-+-{'-'*15}")
        for reg, reg_label in specs:
            res     = adfuller(y, regression=reg, autolag="AIC")
            stat, pval, lags = res[0], res[1], res[2]
            verdict = "stationary" if pval < 0.05 else "unit root"
            print(f"  {reg_label:>20} | {stat:>10.4f} | {pval:>9.4f} | {lags:>5} | {verdict}")

    print()
    res_ref = adfuller(make_i0(), regression="c", autolag="AIC")
    print("  ADF critical values (constant-only specification):")
    for k, v in res_ref[4].items():
        print(f"    {k}: {v:.4f}")
    print()
    print("  Trend-stationary series fails the 'c' ADF but passes 'ct'.")
    print("  Always use 'ct' when the series shows a clear upward or downward drift.")

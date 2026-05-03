"""
Time Series and ARIMA Basics:
- Stationarity: the foundation of time series modeling
- Autocorrelation function (ACF) and partial autocorrelation function (PACF)
- AR(p) models: autoregressive processes and stationarity conditions
- MA(q) models: moving average processes and invertibility
- ARIMA(p, d, q): combining differencing, AR, and MA components
- Estimation with statsmodels: fitting, diagnostics, and model selection
- Forecasting: h-step ahead predictions and confidence intervals
- A practical guide for applied time series modeling
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# We generate four canonical time series processes:
#
#   Random walk:  y_t = y_{t-1} + eps_t  (non-stationary, I(1))
#   AR(1):        y_t = PHI * y_{t-1} + eps_t  (stationary if |PHI| < 1)
#   MA(1):        y_t = eps_t + THETA * eps_{t-1}
#   ARMA(1,1):    y_t = PHI * y_{t-1} + eps_t + THETA * eps_{t-1}
#
# True parameters are known so we can check if the fitted model recovers them.

N_TS    = 300
PHI     = 0.70    # true AR coefficient
THETA   = 0.50    # true MA coefficient
SIGMA   = 1.00    # noise std dev


def make_ar1(phi: float = PHI, n: int = N_TS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y   = np.zeros(n)
    eps = rng.normal(0, SIGMA, n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def make_ma1(theta: float = THETA, n: int = N_TS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, SIGMA, n)
    y   = np.zeros(n)
    for t in range(1, n):
        y[t] = eps[t] + theta * eps[t - 1]
    return y


def make_random_walk(n: int = N_TS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, SIGMA, n)
    return np.cumsum(eps)


def make_arma11(phi: float = PHI, theta: float = THETA,
                n: int = N_TS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, SIGMA, n)
    y   = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t] + theta * eps[t - 1]
    return y


# ==============================================================
# 1. Stationarity
# ==============================================================
# A time series is (weakly) stationary if:
#   1. E[y_t] = μ  (constant mean, no trend)
#   2. Var(y_t) = σ²  (constant variance)
#   3. Cov(y_t, y_{t-k}) depends only on lag k, not on t.
#
# Non-stationary series violate one or more of these conditions.
# The random walk y_t = y_{t-1} + eps_t has:
#   - Var(y_t) = t*σ² → grows without bound.
#   - Cov(y_t, y_{t-k}) = (t-k)*σ² → depends on t, not just k.
#
# Stationarity is required for most time series estimators and
# for any classical inference (t-tests, F-tests) to be valid.
#
# Augmented Dickey-Fuller (ADF) test:
#   H0: series has a unit root (non-stationary).
#   If p < 0.05, reject H0 → stationary.

def demo_stationarity() -> None:
    y_ar  = make_ar1()
    y_rw  = make_random_walk()

    print(f"\n  Stationarity diagnostics  (N = {N_TS})")
    print()
    print(f"  {'Series':>20} | {'Mean':>8} | {'Std':>8} | {'ADF stat':>10} | {'p-value':>9} | Verdict")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-{'-'*15}")

    for label, y in [("AR(1) phi=0.7", y_ar), ("Random walk", y_rw)]:
        adf_res  = adfuller(y, autolag="AIC")
        adf_stat = adf_res[0]
        pval     = adf_res[1]
        verdict  = "stationary" if pval < 0.05 else "non-stationary"
        print(f"  {label:>20} | {y.mean():>8.3f} | {y.std():>8.3f} | "
              f"{adf_stat:>10.3f} | {pval:>9.4f} | {verdict}")

    print()
    adf_diff = adfuller(np.diff(y_rw), autolag="AIC")
    print(f"  First difference of random walk: ADF p = {adf_diff[1]:.4f} (stationary)")
    print()
    print("  The random walk becomes stationary after one differencing (d=1).")
    print("  This is the 'I(1)' designation: integrated of order 1.")
    print()
    print("  Rule of thumb: difference until ADF rejects unit root.")
    print("  Over-differencing (too many d) also causes problems -- add MA terms.")


# ==============================================================
# 2. ACF and PACF
# ==============================================================
# ACF (Autocorrelation Function):
#   ρ(k) = Corr(y_t, y_{t-k})
#   For AR(p): tails off geometrically.
#   For MA(q): cuts off sharply after lag q.
#
# PACF (Partial Autocorrelation Function):
#   φ(k,k) = Corr(y_t, y_{t-k} | y_{t-1}, ..., y_{t-k+1})
#   Removes the linear influence of intermediate lags.
#   For AR(p): cuts off after lag p.
#   For MA(q): tails off geometrically.
#
# Pattern summary table:
#   Model    | ACF pattern        | PACF pattern
#   AR(p)    | tails off          | cuts off at lag p
#   MA(q)    | cuts off at lag q  | tails off
#   ARMA(p,q)| both tail off      | both tail off

def demo_acf_pacf() -> None:
    y_ar  = make_ar1()
    y_ma  = make_ma1()
    n_lags = 10

    print(f"\n  ACF and PACF at lags 1-{n_lags}")
    print()

    for label, y in [("AR(1) phi=0.7", y_ar), ("MA(1) theta=0.5", y_ma)]:
        acf_vals  = acf( y, nlags=n_lags, fft=True)[1:]   # skip lag 0 (always 1)
        pacf_vals = pacf(y, nlags=n_lags)[1:]

        print(f"  {label}")
        print(f"  {'Lag':>5} | {'ACF':>8} | PACF")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}")
        for k in range(n_lags):
            print(f"  {k+1:>5} | {acf_vals[k]:>8.4f} | {pacf_vals[k]:>8.4f}")
        print()

    print("  AR(1): ACF decays geometrically; PACF cuts off after lag 1.")
    print("  MA(1): ACF cuts off after lag 1; PACF decays geometrically.")
    print("  These patterns guide the choice of p and q in ARIMA(p, d, q).")

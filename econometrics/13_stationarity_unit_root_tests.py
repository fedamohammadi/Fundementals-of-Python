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


# ==============================================================
# 3. KPSS Test
# ==============================================================
# KPSS (Kwiatkowski-Phillips-Schmidt-Shin, 1992) reverses the hypothesis:
#   H0: series is stationary.
#   H1: series has a unit root.
#
# Combining ADF and KPSS gives stronger evidence than either alone:
#   ADF rejects AND KPSS fails to reject  ->  stationary (strong evidence)
#   ADF fails to reject AND KPSS rejects  ->  unit root (strong evidence)
#   Both reject OR both fail to reject    ->  conflicting; investigate further
#
# KPSS stat is based on partial sums of OLS residuals; large values = non-stationary.

def demo_kpss_test() -> None:
    series = {
        "I(0) AR(1)":       make_i0(),
        "I(1) random walk": make_i1(),
        "Trend-stat.":      make_trend_stationary(),
    }

    print(f"\n  KPSS test  (H0: stationary, H1: unit root)")
    print()
    print(f"  {'Series':>20} | {'Spec':>4} | {'Stat':>8} | {'5% CV':>8} | Verdict")
    print(f"  {'-'*20}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*22}")

    for label, y in series.items():
        for reg in ["c", "ct"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, pval, lags, crits = kpss(y, regression=reg, nlags="auto")
            cv5     = crits["5%"]
            rejects = stat > cv5
            verdict = "unit root (reject H0)" if rejects else "stationary"
            print(f"  {label:>20} | {reg:>4} | {stat:>8.4f} | {cv5:>8.4f} | {verdict}")

    print()
    print("  ADF vs KPSS decision matrix:")
    print(f"  {'ADF':>18} | {'KPSS':>18} | Conclusion")
    print(f"  {'-'*18}-+-{'-'*18}-+-{'-'*25}")
    for adf, kps, concl in [
        ("rejects H0",      "fails to reject", "stationary (strong)"),
        ("fails to reject", "rejects H0",      "unit root (strong)"),
        ("rejects H0",      "rejects H0",      "conflicting -- investigate"),
        ("fails to reject", "fails to reject", "conflicting -- investigate"),
    ]:
        print(f"  {adf:>18} | {kps:>18} | {concl}")


# ==============================================================
# 4. Phillips-Perron Test
# ==============================================================
# The Phillips-Perron (PP) test shares the same H0/H1 as ADF but uses a
# nonparametric HAC correction instead of augmenting with lagged differences.
#
# Advantage: no need to choose augmenting lags; robust to MA errors.
# Disadvantage: worse size distortions in small samples.
#
# statsmodels lacks a built-in PP implementation, so we approximate it
# via OLS on Δy_t ~ y_{t-1} with HAC (Newey-West) standard errors.
# The t-stat on y_{t-1} is the PP-type statistic.

def _pp_tstat(y: np.ndarray, trend: str = "c") -> float:
    """Return HAC-corrected t-stat for the unit-root coefficient."""
    n   = len(y)
    dy  = np.diff(y)
    yl  = y[:-1]
    if trend == "c":
        X = sm.add_constant(yl)
    else:
        X = np.column_stack([np.ones(len(yl)), np.arange(len(yl)), yl])
    max_lag = int(4 * (n / 100) ** 0.25)
    res = sm.OLS(dy, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})
    return float(res.tvalues[-1])


def demo_pp_test() -> None:
    series = {
        "I(0) AR(1)":       make_i0(),
        "I(1) random walk": make_i1(),
    }

    print(f"\n  Phillips-Perron test  (HAC-corrected DF, H0: unit root)")
    print()
    print(f"  {'Series':>20} | {'ADF stat':>10} | {'ADF p':>7} | {'PP t-stat':>10} | PP verdict")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*7}-+-{'-'*10}-+-{'-'*15}")

    for label, y in series.items():
        adf_stat, adf_p = adfuller(y, regression="c", autolag="AIC")[:2]
        pp_t = _pp_tstat(y, trend="c")
        pp_verdict = "stationary" if pp_t < -2.86 else "unit root"
        print(f"  {label:>20} | {adf_stat:>10.4f} | {adf_p:>7.4f} | {pp_t:>10.4f} | {pp_verdict}")

    print()
    print("  PP and ADF should agree in most cases.")
    print("  Prefer PP when data are seasonally adjusted (likely MA errors).")
    print("  Prefer ADF in small samples with well-motivated lag choice.")
    print("  If they disagree, use KPSS as a tiebreaker.")


# ==============================================================
# 5. Deterministic vs. Stochastic Trends
# ==============================================================
# Two sources of trending behavior require different treatments:
#
#   Stochastic trend (I(1)):  y_t = y_{t-1} + eps_t
#     Variance grows with t; shocks have permanent effects.
#     Treatment: first-difference.
#
#   Deterministic trend (trend-stationary):  y_t = a + b*t + eps_t
#     Variance is constant; shocks are transitory.
#     Treatment: detrend (OLS residual from y ~ 1 + t), not differencing.
#
# Confusing the two:
#   Differencing a trend-stationary series introduces an MA unit root.
#   Detrending an I(1) series leaves non-stationary residuals.

def demo_deterministic_vs_stochastic() -> None:
    rng  = np.random.default_rng(7)
    n    = N
    t    = np.arange(n)
    y_ts = 0.2 * t + rng.normal(0, 1.5, n)          # trend-stationary
    y_i1 = np.cumsum(rng.normal(0, 1, n))            # I(1) random walk

    X_t = sm.add_constant(t.astype(float))
    resid_ts = sm.OLS(y_ts, X_t).fit().resid.values
    resid_i1 = sm.OLS(y_i1, X_t).fit().resid.values

    def adf_p(arr: np.ndarray) -> float:
        return adfuller(arr, regression="c", autolag="AIC")[1]

    print(f"\n  Detrending vs. differencing  (N = {n})")
    print()
    print(f"  {'Series':>22} | {'Original p':>11} | {'Detrended p':>12} | {'Diff p':>8}")
    print(f"  {'-'*22}-+-{'-'*11}-+-{'-'*12}-+-{'-'*8}")
    for label, y, detrended in [("Trend-stationary", y_ts, resid_ts),
                                  ("I(1) random walk",  y_i1, resid_i1)]:
        print(f"  {label:>22} | {adf_p(y):>11.4f} | {adf_p(detrended):>12.4f} | {adf_p(np.diff(y)):>8.4f}")

    print()
    print("  Trend-stationary: detrending achieves stationarity; differencing over-adjusts.")
    print("  I(1) random walk: detrending fails; differencing succeeds.")
    print("  The ADF 'ct' specification includes a trend and helps distinguish the two.")


# ==============================================================
# 6. Seasonal Integration
# ==============================================================
# Seasonal data may have regular AND seasonal unit roots.
#
#   Regular unit root:   level drifts over time.
#   Seasonal unit root:  seasonal pattern itself is non-stationary.
#
# For quarterly data (s=4), a seasonal difference is:
#   Δ₄y_t = y_t - y_{t-4}
#
# Heuristic identification:
#   1. ACF of levels: slow decay everywhere (regular unit root).
#   2. ACF of Δy: slow decay at seasonal lags 4, 8, 12...
#      -> suggests a seasonal unit root remains.
#   3. ACF of Δ₄y: should look like white noise if both removed.
#
# Full seasonal unit root testing (HEGY) is outside this file's scope.

def demo_seasonal_integration() -> None:
    y       = make_seasonal_i1()
    y_diff  = np.diff(y)
    y_sdiff = y[4:] - y[:-4]       # seasonal difference lag-4
    n_lags  = 16
    ci_95   = 1.96 / np.sqrt(N)

    def show_acf_line(label: str, arr: np.ndarray) -> None:
        vals      = acf(arr, nlags=n_lags, fft=True)[1:]
        seas_idx  = {3, 7, 11, 15}    # 0-indexed for lags 4, 8, 12, 16
        peak_seas = max(abs(vals[i]) for i in seas_idx)
        print(f"  {label:>32}: max seasonal-lag |ACF| = {peak_seas:.3f}  "
              f"({'significant' if peak_seas > ci_95 else 'not significant'})")

    print(f"\n  Seasonal integration diagnostics  (quarterly I(1) + seasonality, N={N})")
    print(f"  Threshold for significance: 1.96/√N = {ci_95:.3f}")
    print()
    show_acf_line("Levels", y)
    show_acf_line("First difference Δy", y_diff)
    show_acf_line("Seasonal difference Δ₄y", y_sdiff)

    print()
    adf_p_level = adfuller(y,       regression="c", autolag="AIC")[1]
    adf_p_diff  = adfuller(y_diff,  regression="c", autolag="AIC")[1]
    adf_p_sdiff = adfuller(y_sdiff, regression="c", autolag="AIC")[1]
    print(f"  ADF p-values:  levels = {adf_p_level:.4f}  |  Δy = {adf_p_diff:.4f}  |  Δ₄y = {adf_p_sdiff:.4f}")
    print()
    print("  If Δ₄y is stationary but Δy is not, use SARIMA(p,d,q)(P,D,Q)[4].")
    print("  Adding a seasonal difference (D=1) removes the seasonal stochastic trend.")


# ==============================================================
# 7. Practical Decision Framework
# ==============================================================

def demo_practical_guide() -> None:
    steps = [
        ("Step 1: Plot the series.",
         "Look for drift, growing variance, or seasonal swings.",
         "A drifting or expanding series is likely non-stationary."),

        ("Step 2: Run ADF with the right specification.",
         "No visible trend -> use 'c'.  Visible trend -> use 'ct'.",
         "Non-stationary under 'c' but stationary under 'ct' = trend-stationary."),

        ("Step 3: Confirm with KPSS (reversed null).",
         "ADF rejects AND KPSS fails to reject -> stationary; proceed.",
         "Both point to unit root -> first-difference and retest."),

        ("Step 4: Choose detrending vs. differencing.",
         "Stochastic trend (I(1)): first-difference (set d=1 in ARIMA).",
         "Deterministic trend: regress on time and model OLS residuals."),

        ("Step 5: Handle seasonality.",
         "Check ACF at seasonal lags after regular differencing.",
         "Persistent seasonal ACF -> add seasonal differencing (D=1 in SARIMA)."),

        ("Step 6: Document results.",
         "Report ADF stat, p-value, lag order used, and regression specification.",
         "Note when tests conflict; run sensitivity checks under alternative specs."),

        ("Common mistakes:",
         "Relying on only one test (ADF alone has low power in small samples).",
         "Assuming all macro series are I(1) without testing."),
    ]

    print()
    for question, opt_a, opt_b in steps:
        print(f"  {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  Stationarity testing is the prerequisite for ARIMA, cointegration,")
    print("  and most time series estimators.  Misclassifying the integration order")
    print("  invalidates all downstream inference.  Test carefully.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Unit Roots and Integration Orders")
    demo_integration_orders()

    section("2. Augmented Dickey-Fuller Test")
    demo_adf_test()

    section("3. KPSS Test")
    demo_kpss_test()

    section("4. Phillips-Perron Test")
    demo_pp_test()

    section("5. Deterministic vs. Stochastic Trends")
    demo_deterministic_vs_stochastic()

    section("6. Seasonal Integration")
    demo_seasonal_integration()

    section("7. Practical Decision Framework")
    demo_practical_guide()


if __name__ == "__main__":
    main()

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


# ==============================================================
# 3. AR(p) Models
# ==============================================================
# An AR(p) process models y_t as a linear function of its own past p values:
#
#   y_t = c + phi_1*y_{t-1} + phi_2*y_{t-2} + ... + phi_p*y_{t-p} + eps_t
#
# Stationarity condition:
#   The roots of 1 - phi_1*L - phi_2*L² - ... - phi_p*Lp = 0 must all
#   lie outside the unit circle in the complex plane.
#   For AR(1): |phi_1| < 1.
#
# Long-run variance of a stationary AR(1):  σ_y² = σ²/(1 - phi²)
# Autocorrelation decays geometrically:     ρ(k) = phi^k

def demo_ar_models() -> None:
    y = make_ar1()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y, order=(1, 0, 0)).fit()

    phi_hat = res.params["ar.L1"]
    c_hat   = res.params.get("const", 0.0)

    print(f"\n  AR(1) estimation  (true phi = {PHI:.2f}, N = {N_TS})")
    print()
    print(f"  {'Parameter':>12} | {'True':>7} | {'Estimate':>9} | {'SE':>8} | {'t':>7}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for name, true_val, param_key in [("const", 0.0, "const"),
                                       ("phi_1", PHI, "ar.L1")]:
        if param_key in res.params.index:
            est = res.params[param_key]
            se  = res.bse[param_key]
            t   = est / se
            print(f"  {name:>12} | {true_val:>7.3f} | {est:>9.4f} | {se:>8.4f} | {t:>7.2f}")

    # Stationarity check: |phi| < 1
    print()
    print(f"  Stationarity check: |phi_hat| = {abs(phi_hat):.4f} < 1  -->  stationary")
    print()

    # Long-run variance
    lrv_true = SIGMA ** 2 / (1 - PHI ** 2)
    lrv_est  = SIGMA ** 2 / (1 - phi_hat ** 2)
    print(f"  Long-run variance:  true = {lrv_true:.4f}   estimated = {lrv_est:.4f}")
    print(f"  Sample variance:    {y.var():.4f}")
    print()
    print("  Theoretical ACF decays as phi^k:")
    print(f"  {'Lag k':>8} | {'ACF = phi^k':>12} | Approx expected correlation")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*28}")
    for k in [1, 2, 5, 10]:
        print(f"  {k:>8} | {PHI**k:>12.4f} | {'geometrically decaying' if k > 1 else 'strong, direct effect'}")


# ==============================================================
# 4. MA(q) Models
# ==============================================================
# An MA(q) process models y_t as a linear combination of current and past errors:
#
#   y_t = mu + eps_t + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}
#
# Key properties:
#   - Always stationary (error terms are iid, so variance is constant).
#   - ACF is non-zero only for lags 1 through q, then exactly zero.
#   - PACF tails off geometrically.
#
# Invertibility condition:
#   The MA polynomial must have roots outside the unit circle.
#   For MA(1): |theta| < 1.  This ensures a unique representation.
#   Without invertibility, the MA and AR representations overlap ambiguously.

def demo_ma_models() -> None:
    y = make_ma1()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y, order=(0, 0, 1)).fit()

    theta_hat = res.params["ma.L1"]
    acf_vals  = acf(y, nlags=5, fft=True)

    print(f"\n  MA(1) estimation  (true theta = {THETA:.2f}, N = {N_TS})")
    print()
    print(f"  {'Parameter':>12} | {'True':>7} | {'Estimate':>9} | {'SE':>8} | {'t':>7}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for name, true_val, param_key in [("const", 0.0, "const"),
                                       ("theta_1", THETA, "ma.L1")]:
        if param_key in res.params.index:
            est = res.params[param_key]
            se  = res.bse[param_key]
            t   = est / se
            print(f"  {name:>12} | {true_val:>7.3f} | {est:>9.4f} | {se:>8.4f} | {t:>7.2f}")

    print()
    print(f"  Invertibility check: |theta_hat| = {abs(theta_hat):.4f} < 1  -->  invertible")
    print()
    print("  Theoretical ACF of MA(1):  rho(1) = theta/(1+theta²),  rho(k≥2) = 0")
    rho1_true = THETA / (1 + THETA ** 2)
    print(f"    rho(1) theoretical = {rho1_true:.4f}")
    print(f"    rho(1) sample ACF  = {acf_vals[1]:.4f}")
    print(f"    rho(2) sample ACF  = {acf_vals[2]:.4f}  (should be near 0)")
    print()
    print("  The ACF cuts off sharply after lag q=1, a hallmark of an MA process.")


# ==============================================================
# 5. ARIMA(p, d, q)
# ==============================================================
# ARIMA generalizes ARMA to handle non-stationary series:
#
#   ARIMA(p, d, q): apply an ARMA(p,q) model to the d-th difference of y.
#
#   d=0: y_t is already stationary -> ARMA(p, q)
#   d=1: model Δy_t = y_t - y_{t-1} with ARMA(p, q)
#   d=2: model Δ²y_t = (y_t - y_{t-1}) - (y_{t-2} - y_{t-1}) with ARMA(p, q)
#
# Selecting p, d, q:
#   1. Determine d from ADF tests and ACF plots.
#   2. Inspect ACF and PACF of the d-th differenced series to identify p and q.
#   3. Fit candidate models and compare by AIC or BIC (lower is better).
#
# Special cases:
#   ARIMA(0,1,0): random walk
#   ARIMA(1,0,0): AR(1)
#   ARIMA(0,0,1): MA(1)
#   ARIMA(p,1,q): common choice for economic time series with stochastic trend

def demo_arima() -> None:
    y_rw = make_random_walk()   # I(1) process

    # Compare models by AIC/BIC on the random walk
    candidates = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1)]

    print(f"\n  Model selection on a random walk (ARIMA(0,1,0) is true model)")
    print()
    print(f"  {'Order (p,d,q)':>14} | {'AIC':>10} | {'BIC':>10} | {'Log-lik':>10}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results = {}
    for order in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y_rw, order=order).fit()
        results[order] = res
        print(f"  {str(order):>14} | {res.aic:>10.3f} | {res.bic:>10.3f} | {res.llf:>10.3f}")

    best_aic = min(results, key=lambda o: results[o].aic)
    best_bic = min(results, key=lambda o: results[o].bic)
    print()
    print(f"  Best by AIC: ARIMA{best_aic}")
    print(f"  Best by BIC: ARIMA{best_bic}")
    print()
    print("  BIC penalizes complexity more heavily (larger k penalty = ln(n) vs 2).")
    print("  BIC tends to prefer simpler models; AIC may overfit in small samples.")
    print("  Both should agree with the true order on large enough samples.")


# ==============================================================
# 6. Estimation with statsmodels
# ==============================================================
# statsmodels.tsa.arima.model.ARIMA fits ARIMA models by maximum likelihood.
#
# Key outputs from the fitted result:
#   .params         - estimated coefficients
#   .bse            - standard errors
#   .aic, .bic      - information criteria
#   .resid          - model residuals
#   .plot_diagnostics() - residual plots (requires matplotlib)
#
# Residual diagnostics:
#   - ACF of residuals should show no significant autocorrelation.
#   - Ljung-Box test: H0 = residuals are white noise.
#     If p > 0.05, residuals are not significantly autocorrelated (good).

def demo_estimation() -> None:
    y = make_arma11()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y, order=(1, 0, 1)).fit()

    print(f"\n  ARIMA(1,0,1) fit on ARMA(1,1) data  (true phi={PHI}, theta={THETA})")
    print()
    print(f"  AIC = {res.aic:.3f}   BIC = {res.bic:.3f}   Log-lik = {res.llf:.3f}")
    print()
    print(f"  {'Parameter':>12} | {'True':>7} | {'Estimate':>9} | {'SE':>8} | 95% CI")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}-+-{'-'*22}")

    param_map = [("const", 0.0), ("ar.L1", PHI), ("ma.L1", THETA)]
    for name, true_val in param_map:
        if name in res.params.index:
            est = res.params[name]
            se  = res.bse[name]
            lo  = est - 1.96 * se
            hi  = est + 1.96 * se
            print(f"  {name:>12} | {true_val:>7.3f} | {est:>9.4f} | {se:>8.4f} | [{lo:.3f}, {hi:.3f}]")

    # Ljung-Box test on residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    lb_pval = float(lb["lb_pvalue"].iloc[0])
    print()
    print(f"  Residual diagnostics:")
    print(f"    Ljung-Box test (10 lags): p = {lb_pval:.4f}", end="  ")
    print("(white noise -- good)" if lb_pval > 0.05 else "(autocorrelation detected)")
    print(f"    Residual mean:  {res.resid.mean():.4f}")
    print(f"    Residual std:   {res.resid.std():.4f}  (true sigma = {SIGMA:.2f})")
    print()
    print("  Residuals should look like white noise.  If the Ljung-Box test")
    print("  rejects, the model is under-specified -- increase p or q.")


# ==============================================================
# 7. Forecasting
# ==============================================================
# Once an ARIMA model is fitted, it produces h-step ahead forecasts.
#
# For a fitted ARIMA(p, d, q):
#   - 1-step forecast: plug in last p observed values and last q errors.
#   - h-step forecast (h > 1): recursively substitute predicted values;
#     forecast uncertainty grows with h (confidence intervals widen).
#
# Forecast evaluation:
#   RMSE = sqrt(mean((y_actual - y_hat)²))  -- penalizes large errors more
#   MAE  = mean(|y_actual - y_hat|)          -- more robust to outliers
#
# A good forecasting model should outperform the naive forecast
# (predicting last observed value for all horizons).

def demo_forecasting() -> None:
    y_all = make_ar1(n=N_TS + 20)  # extra 20 observations as hold-out
    y_train = y_all[:N_TS]
    y_test  = y_all[N_TS:]
    h       = len(y_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_ar1  = ARIMA(y_train, order=(1, 0, 0)).fit()
        res_rw   = ARIMA(y_train, order=(0, 1, 0)).fit()

    fc_ar1 = res_ar1.get_forecast(steps=h)
    fc_rw  = res_rw.get_forecast(steps=h)

    # Naive forecast: repeat last observed value
    naive_fc = np.full(h, y_train[-1])

    def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(((actual - pred) ** 2).mean()))

    def mae(actual: np.ndarray, pred: np.ndarray) -> float:
        return float(np.abs(actual - pred).mean())

    print(f"\n  {h}-step ahead forecasting on AR(1) series (true phi = {PHI})")
    print(f"  Train: {N_TS} obs   Test (hold-out): {h} obs")
    print()
    print(f"  {'Model':>22} | {'RMSE':>8} | MAE")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}")

    for label, pred in [("AR(1) forecast",    fc_ar1.predicted_mean),
                         ("Random walk (I(1))", fc_rw.predicted_mean),
                         ("Naive (last value)", naive_fc)]:
        print(f"  {label:>22} | {rmse(y_test, pred):>8.4f} | {mae(y_test, pred):.4f}")

    # Show confidence intervals for AR(1) forecast
    ci = fc_ar1.conf_int(alpha=0.05)
    print()
    print(f"  AR(1) 95% forecast intervals (first 5 steps):")
    print(f"  {'Step':>6} | {'Point':>8} | {'Lower':>8} | Upper")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for i in range(min(5, h)):
        pt = float(fc_ar1.predicted_mean.iloc[i])
        lo = float(ci.iloc[i, 0])
        hi = float(ci.iloc[i, 1])
        print(f"  {i+1:>6} | {pt:>8.4f} | {lo:>8.4f} | {hi:.4f}")

    print()
    print("  Confidence intervals widen with horizon -- uncertainty grows as")
    print("  the forecast relies on more predicted (rather than observed) values.")
    print("  An AR(1) model should outperform the naive forecast when |phi| > 0.")


# ==============================================================
# 8. Practical Guide
# ==============================================================

def demo_practical_guide() -> None:
    guide = [
        ("Step 1: Check for stationarity.",
         "Plot the series.  If it drifts or variance grows, it is likely non-stationary.",
         "Run the ADF test.  If p > 0.05, difference the series (increase d by 1)."),

        ("Step 2: Identify p and q from ACF/PACF of the differenced series.",
         "PACF cuts off at lag p  -> AR(p) component.",
         "ACF cuts off at lag q   -> MA(q) component."),

        ("Step 3: Fit candidate models and compare AIC/BIC.",
         "Start simple: AR(1), MA(1), ARMA(1,1).",
         "Fit a small grid of (p, d, q) values and select the minimum AIC or BIC."),

        ("Step 4: Check residuals.",
         "ACF of residuals should show no significant spikes (white noise).",
         "Ljung-Box p > 0.05 means residuals are not significantly autocorrelated."),

        ("Step 5: Forecast and evaluate.",
         "Use get_forecast() for point predictions and confidence intervals.",
         "Evaluate on a held-out test set with RMSE and MAE."),

        ("Common pitfalls:",
         "Over-differencing: too many d introduces MA structure that wasn't there.",
         "Ignoring seasonality: use SARIMA for monthly/quarterly economic data."),

        ("When ARIMA is not enough:",
         "Structural breaks: use regime-switching or segmented trend models.",
         "Multivariate dynamics: use VAR (vector autoregression) across series."),
    ]

    print()
    for i, (question, opt_a, opt_b) in enumerate(guide, 1):
        print(f"  {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  ARIMA is the workhorse of univariate time series econometrics.")
    print("  It handles trends via differencing and short-run dynamics via AR/MA.")
    print("  For seasonality, cointegration, or multivariate series, see files 13-16.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Stationarity")
    demo_stationarity()

    section("2. ACF and PACF")
    demo_acf_pacf()

    section("3. AR(p) Models")
    demo_ar_models()

    section("4. MA(q) Models")
    demo_ma_models()

    section("5. ARIMA(p, d, q)")
    demo_arima()

    section("6. Estimation with statsmodels")
    demo_estimation()

    section("7. Forecasting")
    demo_forecasting()

    section("8. Practical Guide")
    demo_practical_guide()


if __name__ == "__main__":
    main()

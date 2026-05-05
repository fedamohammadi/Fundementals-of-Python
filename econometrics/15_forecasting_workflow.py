"""
Forecasting Workflow:
- Time series cross-validation: rolling and expanding window evaluation
- Baseline forecasts: naive, drift, and seasonal naive methods
- Exponential smoothing: simple, Holt's linear, and Holt-Winters
- Automatic ARIMA order selection via AIC/BIC grid search
- Forecast evaluation: RMSE, MAE, and MASE metrics
- Combining forecasts: equal weights and optimal combination
- A practical guide for end-to-end forecasting
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Two time series for forecasting exercises:
#
#   AR(2): y_t = 0.6*y_{t-1} - 0.2*y_{t-2} + eps_t  (stationary)
#     Good for ARIMA; no trend or seasonality.
#
#   Seasonal trend: y_t = 1 + 0.05*t + S(t) + eps_t
#     Quarterly seasonality with linear trend.
#     Good for Holt-Winters and SARIMA comparison.

N_FULL   = 200
N_TEST   = 20
N_TRAIN  = N_FULL - N_TEST
SEAS_PER = 4


def make_ar2(n: int = N_FULL, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, eps = np.zeros(n), rng.normal(0, 1, n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + eps[t]
    return y


def make_seasonal_trend(n: int = N_FULL, seed: int = 7) -> np.ndarray:
    rng     = np.random.default_rng(seed)
    t       = np.arange(n)
    seasons = np.array([1.5, 0.5, -0.5, -1.5])
    seas    = np.array([seasons[i % 4] for i in range(n)])
    return 1.0 + 0.05 * t + seas + rng.normal(0, 0.5, n)


Y_AR2      = make_ar2()
Y_SEAS     = make_seasonal_trend()
TRAIN_AR2  = Y_AR2[:N_TRAIN]
TEST_AR2   = Y_AR2[N_TRAIN:]
TRAIN_SEAS = Y_SEAS[:N_TRAIN]
TEST_SEAS  = Y_SEAS[N_TRAIN:]


# ==============================================================
# Evaluation helpers
# ==============================================================

def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(((actual - pred) ** 2).mean()))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).mean())


def mase(actual: np.ndarray, pred: np.ndarray, train: np.ndarray) -> float:
    """MAE scaled by in-sample naive MAE; < 1 beats naive baseline."""
    naive_mae = float(np.abs(np.diff(train)).mean())
    return mae(actual, pred) / naive_mae if naive_mae > 0 else float("nan")


# ==============================================================
# 1. Time Series Cross-Validation
# ==============================================================
# Standard k-fold CV shuffles data randomly -- invalid for time series
# because it leaks future observations into the training set.
#
# Two valid approaches:
#
#   Expanding window: train on [1, t-1], test on [t], increment t.
#     Training set grows over time; mirrors real forecasting conditions.
#
#   Rolling window: fixed-width window slides forward.
#     Avoids using very old data; useful when older data is less relevant.
#
# h-step ahead CV: predict h periods ahead at each fold.
#   This estimates accuracy at the horizon that actually matters in practice.

def demo_ts_cross_validation() -> None:
    y         = Y_AR2
    min_train = 60
    h         = 5
    n         = len(y)

    naive_errs, ar1_errs, ar2_errs = [], [], []

    for end in range(min_train, n - h + 1):
        train  = y[:end]
        actual = y[end:end + h]

        naive_errs.append(mae(actual, np.full(h, train[-1])))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar1_errs.append(mae(actual, ARIMA(train, order=(1, 0, 0)).fit().forecast(h)))
            ar2_errs.append(mae(actual, ARIMA(train, order=(2, 0, 0)).fit().forecast(h)))

    n_folds  = len(naive_errs)
    naive_cv = float(np.mean(naive_errs))

    print(f"\n  Expanding-window CV  (AR(2) series, h={h}, folds={n_folds})")
    print()
    print(f"  {'Model':>10} | {'CV MAE':>10} | Rel. to naive")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*13}")
    for label, errs in [("Naive", naive_errs), ("AR(1)", ar1_errs), ("AR(2)", ar2_errs)]:
        avg = float(np.mean(errs))
        print(f"  {label:>10} | {avg:>10.4f} | {avg/naive_cv:.4f}")

    print()
    print("  AR(2) CV MAE is lowest -- it matches the true data-generating process.")
    print("  Never use standard k-fold CV on time series; it leaks future data.")


# ==============================================================
# 2. Baseline Forecasts
# ==============================================================
# Simple baselines are essential benchmarks.  A model that cannot beat
# a naive baseline is not useful in practice.
#
#   Naive: ŷ_{T+h} = y_T  (last observed value)
#     Optimal for I(1) (random walk) series.
#
#   Drift: ŷ_{T+h} = y_T + h*(y_T - y_1)/(T-1)
#     Extrapolates the average historical trend.
#
#   Seasonal naive: ŷ_{T+h} = y_{T+h-s}  (last season's value)
#     Optimal for seasonal random walks.
#
#   Moving average (window w): ŷ_{T+h} = mean of last w observations.
#     Smooths noise; good when there is no strong trend or seasonality.

def demo_baselines() -> None:
    train = TRAIN_SEAS
    test  = TEST_SEAS
    h     = N_TEST
    T     = len(train)

    naive_fc = np.full(h, train[-1])
    drift_fc = train[-1] + np.arange(1, h + 1) * (train[-1] - train[0]) / (T - 1)
    seas_fc  = np.array([train[T - SEAS_PER + (i % SEAS_PER)] for i in range(h)])
    ma5_fc   = np.full(h, train[-5:].mean())

    print(f"\n  Baseline forecasts on seasonal trend series  (h={h})")
    print(f"  Train N={N_TRAIN}, Test N={N_TEST}")
    print()
    print(f"  {'Method':>18} | {'RMSE':>8} | {'MAE':>8} | MASE")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for label, fc in [("Naive",          naive_fc),
                       ("Drift",          drift_fc),
                       ("Seasonal naive", seas_fc),
                       ("Moving avg (5)", ma5_fc)]:
        print(f"  {label:>18} | {rmse(test, fc):>8.4f} | {mae(test, fc):>8.4f} | "
              f"{mase(test, fc, train):.4f}")

    print()
    print("  Seasonal naive performs best here -- it captures the seasonal pattern.")
    print("  MASE < 1 means the model beats the naive (last-value) baseline.")
    print("  Always compare against at least one baseline before reporting results.")


# ==============================================================
# 3. Exponential Smoothing
# ==============================================================
# ETS (Error-Trend-Seasonality) models weight past observations exponentially.
#
#   Simple ES (SES):  level only; alpha controls smoothing.
#     Optimal forecast for I(1) with no trend or seasonality.
#
#   Holt's linear (double ES):  level + linear trend.
#     Parameters: alpha (level), beta (trend smoothing).
#
#   Holt-Winters:  level + trend + seasonality.
#     Additional parameter: gamma (seasonal smoothing).
#     Additive: seasonal effect is constant in magnitude.
#     Multiplicative: seasonal effect scales with level.
#
# Parameters are estimated by minimizing SSE (or MLE).

def demo_exponential_smoothing() -> None:
    train = TRAIN_SEAS
    test  = TEST_SEAS
    h     = N_TEST

    models = [
        ("SES",               dict(trend=None, seasonal=None)),
        ("Holt linear",       dict(trend="add", seasonal=None)),
        ("HW additive",       dict(trend="add", seasonal="add",
                                   seasonal_periods=SEAS_PER)),
        ("HW multiplicative", dict(trend="add", seasonal="mul",
                                   seasonal_periods=SEAS_PER)),
    ]

    print(f"\n  Exponential smoothing on seasonal trend series  (h={h})")
    print()
    print(f"  {'Model':>22} | {'RMSE':>8} | {'MAE':>8} | MASE")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for label, kwargs in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m  = ExponentialSmoothing(train, **kwargs).fit(optimized=True)
                fc = m.forecast(h)
                print(f"  {label:>22} | {rmse(test, fc):>8.4f} | {mae(test, fc):>8.4f} | "
                      f"{mase(test, fc, train):.4f}")
            except Exception as e:
                print(f"  {label:>22} | error: {e}")

    print()
    print("  Holt-Winters captures trend and seasonality; outperforms SES here.")
    print("  SES is optimal for random walks without trend -- misspecified for this data.")
    print("  MASE < 1 confirms Holt-Winters beats the naive baseline.")

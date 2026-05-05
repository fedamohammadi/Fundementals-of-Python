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

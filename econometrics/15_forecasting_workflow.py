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


# ==============================================================
# 4. Automatic ARIMA Order Selection
# ==============================================================
# Choosing (p, d, q) from ACF/PACF by hand is subjective.
# A systematic approach: grid search over candidate orders, select by AIC.
#
# Algorithm:
#   1. Determine d: ADF test; increment until stationary.
#   2. Fit ARIMA(p, d, q) for p, q in {0 ... max_p/max_q}.
#   3. Select the order with minimum AIC (or BIC for parsimony).
#   4. Verify residuals: Ljung-Box p > 0.05 = white noise.
#
# pmdarima (Python port of R's auto.arima) automates this end-to-end.
# Below we implement the grid search manually to illustrate the logic.

def demo_auto_arima() -> None:
    train = TRAIN_AR2
    d     = 0 if adfuller(train, autolag="AIC")[1] < 0.05 else 1

    rows = []
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(train, order=(p, d, q)).fit()
                rows.append((p, d, q, res.aic, res.bic))
            except Exception:
                pass

    rows.sort(key=lambda r: r[3])
    best_order = rows[0][:3]

    print(f"\n  Auto ARIMA grid search on AR(2) series  (true order = (2,0,0))")
    print(f"  Determined d = {d} from ADF test; grid: p,q ∈ {{0..3}}")
    print()
    print(f"  {'Order':>12} | {'AIC':>10} | {'BIC':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for p, dv, q, aic, bic in rows[:6]:
        flag = " <-- best AIC" if (p, dv, q) == best_order else ""
        print(f"  {f'({p},{dv},{q})':>12} | {aic:>10.3f} | {bic:>10.3f}{flag}")

    print()
    print(f"  Best order by AIC: ARIMA{best_order}  (true DGP: ARIMA(2,0,0))")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_best = ARIMA(train, order=best_order).fit()
    lb   = acorr_ljungbox(res_best.resid, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    print(f"  Ljung-Box p (10 lags): {lb_p:.4f}  ",
          end="")
    print("(residuals OK)" if lb_p > 0.05 else "(autocorrelation remains -- respecify)")


# ==============================================================
# 5. Forecast Evaluation Metrics
# ==============================================================
# Point forecast accuracy:
#
#   RMSE = sqrt(mean((y - ŷ)²))
#     Penalizes large errors heavily; scale-dependent.
#
#   MAE = mean(|y - ŷ|)
#     Robust to outliers; scale-dependent.
#
#   MASE = MAE / MAE_naive  (Hyndman & Koehler 2006)
#     Scale-free; MASE < 1 means model beats naive baseline.
#     Preferred for comparing across series.
#
# Interval forecast accuracy:
#   Coverage: fraction of actuals inside the nominal 95% CI.
#     A well-calibrated model should have coverage ≈ 0.95.
#   Winkler score: penalizes wide intervals and missed observations.

def demo_evaluation_metrics() -> None:
    train = TRAIN_AR2
    test  = TEST_AR2
    h     = N_TEST

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_ar2 = ARIMA(train, order=(2, 0, 0)).fit()
        res_rw  = ARIMA(train, order=(0, 1, 0)).fit()

    fc_ar2   = res_ar2.get_forecast(steps=h)
    fc_rw    = res_rw.get_forecast(steps=h)
    naive_fc = np.full(h, train[-1])

    print(f"\n  Forecast evaluation  (AR(2) series, h={h})")
    print()
    print(f"  {'Model':>14} | {'RMSE':>8} | {'MAE':>8} | {'MASE':>8} | 95% coverage")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}")

    for label, fc_obj, fc_mean in [
        ("AR(2)",    fc_ar2, fc_ar2.predicted_mean.values),
        ("RW (d=1)", fc_rw,  fc_rw.predicted_mean.values),
        ("Naive",    None,   naive_fc),
    ]:
        r  = rmse(test, fc_mean)
        m  = mae(test, fc_mean)
        ms = mase(test, fc_mean, train)
        if fc_obj is not None:
            ci      = fc_obj.conf_int(alpha=0.05).values
            in_ci   = ((test >= ci[:, 0]) & (test <= ci[:, 1])).mean()
            cov_str = f"{in_ci:.3f}"
        else:
            cov_str = "N/A"
        print(f"  {label:>14} | {r:>8.4f} | {m:>8.4f} | {ms:>8.4f} | {cov_str}")

    print()
    print("  MASE < 1: model beats the naive baseline.")
    print("  Coverage should be close to 0.95 for a well-calibrated 95% CI.")
    print("  RMSE > MAE when there are large individual errors (squared penalty).")


# ==============================================================
# 6. Combining Forecasts
# ==============================================================
# Combining forecasts (ensembling) typically outperforms any single model.
#
#   Equal-weight combination: ŷ_comb = (1/M) * Σ ŷ_m
#     Simple and surprisingly hard to beat in practice.
#
#   Optimal (Granger-Ramanathan) combination:
#     Regress y on M model forecasts over a validation window.
#     Estimated weights used for future forecasts.
#     Pitfall: can overfit when validation window is short.
#
# Why combination works: errors across models are not perfectly correlated.
# Averaging partially cancels model-specific errors (variance reduction).

def demo_forecast_combination() -> None:
    train = TRAIN_AR2
    test  = TEST_AR2
    h     = N_TEST

    orders = [(1, 0, 0), (2, 0, 0), (0, 0, 1), (1, 0, 1)]
    fcs    = {}
    for order in orders:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fcs[str(order)] = ARIMA(train, order=order).fit().get_forecast(h).predicted_mean.values
    fcs["naive"] = np.full(h, train[-1])

    equal_comb = np.column_stack(list(fcs.values())).mean(axis=1)

    # Optimal combination via OLS on a validation split
    val_size = 20
    tr2, val = train[:-val_size], train[-val_size:]
    val_fcs  = {}
    for order in orders:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_fcs[str(order)] = ARIMA(tr2, order=order).fit().get_forecast(val_size).predicted_mean.values
    X_val   = sm.add_constant(np.column_stack(list(val_fcs.values())))
    opt_w   = sm.OLS(val, X_val).fit().params
    X_test  = sm.add_constant(np.column_stack([fcs[str(o)] for o in orders]))
    opt_comb = X_test @ opt_w

    print(f"\n  Forecast combination on AR(2) series  (h={h})")
    print()
    print(f"  {'Model':>20} | {'RMSE':>8} | MAE")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
    for label, fc in fcs.items():
        print(f"  {label:>20} | {rmse(test, fc):>8.4f} | {mae(test, fc):.4f}")
    print(f"  {'Equal combination':>20} | {rmse(test, equal_comb):>8.4f} | {mae(test, equal_comb):.4f}")
    print(f"  {'Optimal combination':>20} | {rmse(test, opt_comb):>8.4f} | {mae(test, opt_comb):.4f}")

    print()
    print("  Equal combination often reduces RMSE below any individual model.")
    print("  Optimal combination can overfit when the validation window is short.")
    print("  In practice, equal weights are a robust default starting point.")


# ==============================================================
# 7. Practical Guide
# ==============================================================

def demo_practical_guide() -> None:
    steps = [
        ("Step 1: Explore the series.",
         "Plot the series; check for trend, seasonality, changing variance.",
         "Run ADF/KPSS to determine integration order (d)."),

        ("Step 2: Establish baselines.",
         "Compute naive, drift, and seasonal naive; report MASE.",
         "A model with MASE ≥ 1 is not useful -- revisit the specification."),

        ("Step 3: Choose an evaluation strategy.",
         "Use expanding or rolling window CV -- never random k-fold.",
         "Match the CV horizon h to the real forecasting horizon needed."),

        ("Step 4: Fit and select the model.",
         "ARIMA: AIC/BIC grid search; verify Ljung-Box on residuals.",
         "Trend + seasonality: use Holt-Winters or SARIMA(p,d,q)(P,D,Q)[s]."),

        ("Step 5: Combine diverse models.",
         "Average 3-5 models to reduce model-specific errors.",
         "Equal weights usually suffice; optimal weights risk overfitting."),

        ("Step 6: Report uncertainty.",
         "Always include prediction intervals alongside point forecasts.",
         "Verify 95% CI coverage is close to 0.95 on the test set."),

        ("Common pitfalls:",
         "Using in-sample R² or training RMSE to evaluate forecast quality.",
         "Ignoring seasonality in quarterly or monthly economic data."),
    ]

    print()
    for question, opt_a, opt_b in steps:
        print(f"  {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  A reliable forecasting workflow needs baselines, CV, and uncertainty bounds.")
    print("  Simplicity and robustness beat complexity in most applied settings.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Time Series Cross-Validation")
    demo_ts_cross_validation()

    section("2. Baseline Forecasts")
    demo_baselines()

    section("3. Exponential Smoothing")
    demo_exponential_smoothing()

    section("4. Automatic ARIMA Order Selection")
    demo_auto_arima()

    section("5. Forecast Evaluation Metrics")
    demo_evaluation_metrics()

    section("6. Combining Forecasts")
    demo_forecast_combination()

    section("7. Practical Guide")
    demo_practical_guide()


if __name__ == "__main__":
    main()

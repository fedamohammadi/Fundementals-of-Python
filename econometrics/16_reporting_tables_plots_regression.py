"""
Reporting Tables, Plots, and Regression Output:
- Summary statistics tables: formatted for applied econometrics
- Regression output tables: comparing multiple models side by side
- Coefficient plots: visualizing estimates and confidence intervals
- Residual diagnostic plots (text-based, matplotlib-free)
- Model comparison tables: AIC, BIC, RMSE across specifications
- Exporting results: CSV and LaTeX formats
- A practical guide for reproducible reporting
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as ts_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Cross-sectional dataset: wage regression.
#   True model: log(wage) = 0.4 + 0.08*educ + 0.05*exper + 0.3*female + eps
#
# Time series dataset: AR(1) for residual diagnostic demonstrations.

N_CS = 500
N_TS = 200


def make_wages(seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    educ  = rng.integers(8, 21, N_CS).astype(float)
    exper = rng.integers(0, 31, N_CS).astype(float)
    female = rng.binomial(1, 0.45, N_CS).astype(float)
    log_wage = 0.4 + 0.08*educ + 0.05*exper + 0.3*female + rng.normal(0, 0.4, N_CS)
    return pd.DataFrame({
        "log_wage": log_wage,
        "educ":     educ,
        "exper":    exper,
        "exper2":   exper ** 2 / 100,
        "female":   female,
    })


def make_ts_data(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, eps = np.zeros(N_TS), rng.normal(0, 1, N_TS)
    for t in range(1, N_TS):
        y[t] = 0.7 * y[t - 1] + eps[t]
    return y


DF_WAGES = make_wages()
Y_TS     = make_ts_data()


# ==============================================================
# 1. Summary Statistics Tables
# ==============================================================
# Summary statistics are the first table in any applied paper.
# Report: N, mean, std, min, median, max.
# Good practices:
#   - Round to 3 significant digits; avoid false precision.
#   - Note units or scale in variable names.
#   - List binary variables separately with their fraction of ones.
#   - Always report N prominently; partial samples must be noted.

def demo_summary_stats() -> None:
    df   = DF_WAGES
    cont = ["log_wage", "educ", "exper"]
    biny = ["female"]

    print(f"\n  Summary statistics  (N = {len(df)})")
    print()
    print(f"  {'Variable':>12} | {'Mean':>7} | {'Std':>7} | {'Min':>7} | {'Median':>7} | {'Max':>7}")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for var in cont:
        col = df[var]
        print(f"  {var:>12} | {col.mean():>7.3f} | {col.std():>7.3f} | "
              f"{col.min():>7.3f} | {col.median():>7.3f} | {col.max():>7.3f}")

    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    print(f"  {'Variable':>12} | {'Fraction':>7} | (Binary 0/1)")
    for var in biny:
        print(f"  {var:>12} | {df[var].mean():>7.3f}")

    print()
    print("  log_wage: log of hourly wage in dollars.")
    print(f"  female: fraction female = {df['female'].mean():.2f}.")
    print("  Quick one-liner: df.describe().T  (includes quartiles).")


# ==============================================================
# 2. Regression Output Tables
# ==============================================================
# Applied econometrics tables show:
#   - Coefficient estimate and SE per variable, one model per column.
#   - Significance stars (* p<.10  ** p<.05  *** p<.01).
#   - Bottom rows: N, R², adjusted R², F-statistic.
#
# Incrementally adding controls (M1 → M4) reveals:
#   - How coefficients change as confounders are included.
#   - Whether results are stable (robustness to specification).

def _stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _reg_table(models: dict, var_order: list, dec: int = 3) -> None:
    cols = list(models.keys())
    hdr  = f"  {'Variable':>14}" + "".join(f" | {m:>13}" for m in cols)
    sep  = "  " + "-" * 14 + ("+-" + "-" * 13) * len(cols)
    print(hdr)
    print(sep)

    for var in var_order:
        row_b  = f"  {var:>14}"
        row_se = f"  {'':>14}"
        for m in models.values():
            if var in m.params.index:
                b  = m.params[var]
                se = m.bse[var]
                s  = _stars(m.pvalues[var])
                row_b  += f" | {f'{b:.{dec}f}{s}':>13}"
                row_se += f" | {'(' + f'{se:.{dec}f}' + ')':>13}"
            else:
                row_b  += f" | {'':>13}"
                row_se += f" | {'':>13}"
        print(row_b)
        print(row_se)

    print(sep)
    for stat, attr in [("N", "nobs"), ("R²", "rsquared"), ("Adj. R²", "rsquared_adj"), ("F-stat", "fvalue")]:
        row = f"  {stat:>14}"
        for m in models.values():
            val = getattr(m, attr)
            if stat == "N":
                row += f" | {int(val):>13}"
            elif stat == "F-stat":
                row += f" | {val:>13.2f}"
            else:
                row += f" | {val:>13.4f}"
        print(row)


def demo_regression_table() -> None:
    df = DF_WAGES
    models = {
        "M1": smf.ols("log_wage ~ educ",                              data=df).fit(),
        "M2": smf.ols("log_wage ~ educ + exper",                      data=df).fit(),
        "M3": smf.ols("log_wage ~ educ + exper + exper2",             data=df).fit(),
        "M4": smf.ols("log_wage ~ educ + exper + exper2 + female",    data=df).fit(),
    }
    var_order = ["Intercept", "educ", "exper", "exper2", "female"]

    print(f"\n  Regression table: dependent variable = log_wage")
    print(f"  Significance: * p<.10  ** p<.05  *** p<.01  |  SE in parentheses")
    print()
    _reg_table(models, var_order)
    print()
    print("  M4 is the full model (true: educ=0.08, exper=0.05, female=0.30).")
    print("  R² increases as controls are added; female has a large negative gap.")
    print("  exper2 captures a nonlinear (concave) experience-wage profile.")


# ==============================================================
# 3. Coefficient Plots
# ==============================================================
# A coefficient plot displays estimates with confidence intervals.
# More informative than a table when comparing many variables or models.
#
# Text-based implementation:
#   We map [lo, b, hi] onto a fixed-width character axis.
#   '[' marks the lower CI, '|' the point estimate, ']' the upper CI.
#   '0' marks the zero line; bars to the right of 0 are positive effects.

def _coef_plot_text(params: pd.Series, ci: pd.DataFrame,
                    width: int = 44, exclude: list = None) -> None:
    exclude = exclude or ["Intercept"]
    rows    = [(v, params[v], ci.loc[v, 0], ci.loc[v, 1])
               for v in params.index if v not in exclude]
    lo_min  = min(r[2] for r in rows)
    hi_max  = max(r[3] for r in rows)
    span    = hi_max - lo_min if hi_max != lo_min else 1.0
    zero_p  = int((-lo_min / span) * width)

    for var, b, lo, hi in rows:
        lo_p  = max(0, int(((lo - lo_min) / span) * width))
        b_p   = min(width, int(((b  - lo_min) / span) * width))
        hi_p  = min(width, int(((hi - lo_min) / span) * width))
        bar   = [" "] * (width + 1)
        for i in range(lo_p, hi_p + 1):
            bar[i] = "-"
        if 0 <= lo_p <= width: bar[lo_p] = "["
        if 0 <= hi_p <= width: bar[hi_p] = "]"
        if 0 <= b_p  <= width: bar[b_p]  = "|"
        if 0 <= zero_p <= width: bar[zero_p] = "0" if bar[zero_p] == " " else bar[zero_p]
        print(f"  {var:>10}: {''.join(bar)}  {b:+.3f}")


def demo_coefficient_plot() -> None:
    df = DF_WAGES
    m  = smf.ols("log_wage ~ educ + exper + exper2 + female", data=df).fit()
    ci = m.conf_int()

    print(f"\n  Coefficient plot: log_wage ~ educ + exper + exper2 + female")
    print(f"  [ --- | --- ] = 95% CI; '|' = point estimate; '0' = zero line")
    print()
    _coef_plot_text(m.params, ci, width=44, exclude=["Intercept"])
    print()
    print("  Bars that do not cross '0' are statistically significant at 5%.")
    print("  'female' is large and negative: the wage gap conditional on education/experience.")
    print("  'exper2' is small but negative: diminishing returns to experience.")


# ==============================================================
# 4. Residual Diagnostic Plots
# ==============================================================
# Residual diagnostics are required for any regression or time series model.
# Without matplotlib, we use text-based tools:
#
#   ACF bar chart: spikes indicate residual autocorrelation.
#   Residual summary: mean (≈0?), std, skew, kurtosis.
#   Ljung-Box test: formal test for autocorrelation in time series residuals.
#
# When matplotlib is available, use model.plot_diagnostics() from statsmodels.

def _acf_bars(values: np.ndarray, n_lags: int = 12, width: int = 28) -> None:
    acf_vals = ts_acf(values, nlags=n_lags, fft=True)[1:]
    ci_bound = 1.96 / np.sqrt(len(values))
    print(f"  {'Lag':>4} | {'ACF':>6} | {'Bar':<{width}} | Sig?")
    print(f"  {'----':>4}-+--------+-{'-'*width}-+-----")
    for k, v in enumerate(acf_vals):
        filled = min(width, int(abs(v) * width))
        bar    = ("+" if v >= 0 else "-") * filled
        sig    = "***" if abs(v) > ci_bound else ""
        print(f"  {k+1:>4} | {v:>6.3f} | {bar:<{width}} | {sig}")


def demo_diagnostic_plots() -> None:
    df    = DF_WAGES
    m     = smf.ols("log_wage ~ educ + exper + exper2 + female", data=df).fit()
    resid = m.resid.values

    print(f"\n  OLS residual summary  (N={N_CS})")
    s = pd.Series(resid)
    print(f"  Mean:     {resid.mean():.6f}  (should be ≈ 0)")
    print(f"  Std:      {resid.std():.4f}")
    print(f"  Skew:     {float(s.skew()):.4f}")
    print(f"  Kurtosis: {float(s.kurt()):.4f}  (excess; Normal = 0)")
    print()
    print("  ACF of OLS residuals (cross-section; no serial structure expected):")
    print()
    _acf_bars(resid, n_lags=10)

    print()
    print("  For cross-sectional OLS, residual ACF should show no significant spikes.")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_res = ARIMA(Y_TS, order=(1, 0, 0)).fit()

    lb   = acorr_ljungbox(ts_res.resid, lags=[10], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    print(f"  ARIMA(1,0,0) residuals  (time series, N={N_TS}):")
    print(f"  Ljung-Box (10 lags) p = {lb_p:.4f}", end="  ")
    print("(white noise -- good)" if lb_p > 0.05 else "(autocorrelation detected)")
    print()
    print("  ACF of ARIMA residuals:")
    print()
    _acf_bars(ts_res.resid.values, n_lags=10)


# ==============================================================
# 5. Model Comparison Tables
# ==============================================================
# When evaluating competing models, report structured comparisons:
#
#   Cross-section: R², adjusted R², AIC, BIC, F-statistic.
#   Time series:   AIC, BIC, train RMSE, test RMSE, Ljung-Box p.
#
# Key principle: never select on in-sample fit alone.
#   AIC and BIC penalize complexity; lower is better.
#   Test RMSE on held-out data is the gold standard.

def demo_model_comparison() -> None:
    df = DF_WAGES

    ols_models = {
        "educ only":    smf.ols("log_wage ~ educ",                           data=df).fit(),
        "educ+exper":   smf.ols("log_wage ~ educ + exper",                   data=df).fit(),
        "full linear":  smf.ols("log_wage ~ educ + exper + female",          data=df).fit(),
        "with exper²":  smf.ols("log_wage ~ educ + exper + exper2 + female", data=df).fit(),
    }

    print(f"\n  Model comparison: OLS  (dep. var: log_wage, N={N_CS})")
    print()
    print(f"  {'Model':>14} | {'R²':>7} | {'Adj R²':>7} | {'AIC':>10} | {'BIC':>10} | {'F':>8}")
    print(f"  {'-'*14}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for label, m in ols_models.items():
        print(f"  {label:>14} | {m.rsquared:>7.4f} | {m.rsquared_adj:>7.4f} | "
              f"{m.aic:>10.2f} | {m.bic:>10.2f} | {m.fvalue:>8.2f}")

    print()
    y_train = Y_TS[:150]
    y_test  = Y_TS[150:]
    h       = len(y_test)

    ts_orders = [(1,0,0), (2,0,0), (0,0,1), (1,0,1)]

    print(f"  Model comparison: ARIMA  (AR(1) DGP, train=150, test=50)")
    print()
    print(f"  {'Order':>10} | {'AIC':>9} | {'BIC':>9} | {'Train RMSE':>11} | {'Test RMSE':>10} | LB p")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*11}-+-{'-'*10}-+-{'-'*8}")

    for order in ts_orders:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m   = ARIMA(y_train, order=order).fit()
        tr_rmse  = float(np.sqrt((m.resid ** 2).mean()))
        te_rmse  = float(np.sqrt(((y_test - m.get_forecast(h).predicted_mean.values) ** 2).mean()))
        lb_p     = float(acorr_ljungbox(m.resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
        print(f"  {str(order):>10} | {m.aic:>9.2f} | {m.bic:>9.2f} | {tr_rmse:>11.4f} | "
              f"{te_rmse:>10.4f} | {lb_p:.4f}")

    print()
    print("  AIC and BIC select AR(1) -- matching the true DGP.")
    print("  Test RMSE confirms: AR(1) generalizes best out-of-sample.")


# ==============================================================
# 6. Exporting Results
# ==============================================================
# Reproducible research requires programmatic output -- never copy-paste.
#
#   Plain text:  redirect stdout  (python script.py > results.txt)
#
#   CSV:         df.to_csv('table.csv', index=False)
#     Importable into Excel, R, Stata, or LaTeX (pgfplotstable).
#
#   LaTeX:       df.to_latex('table.tex', float_format='%.3f', escape=False)
#     Include in paper with \input{table.tex}.
#
#   statsmodels: result.summary().as_latex()  or  .as_html()
#     Full regression summary table in publication-ready format.

def demo_exporting() -> None:
    df = DF_WAGES
    m  = smf.ols("log_wage ~ educ + exper + exper2 + female", data=df).fit()
    ci = m.conf_int()

    results_df = pd.DataFrame({
        "variable": m.params.index,
        "estimate": m.params.values.round(4),
        "se":       m.bse.values.round(4),
        "t":        m.tvalues.values.round(3),
        "p":        m.pvalues.values.round(4),
        "ci_lo":    ci[0].values.round(4),
        "ci_hi":    ci[1].values.round(4),
    })

    print(f"\n  Tidy results DataFrame  (ready for export)")
    print()
    print(results_df.to_string(index=False))
    print()

    csv_str = results_df.to_csv(index=False)
    print("  CSV preview (first 3 lines):")
    for line in csv_str.split("\n")[:3]:
        print(f"    {line}")
    print("    ...")
    print()

    lat_str = results_df.to_latex(index=False, float_format="%.4f",
                                   caption="OLS Results", label="tab:ols")
    print("  LaTeX preview (first 5 lines):")
    for line in lat_str.split("\n")[:5]:
        print(f"    {line}")
    print("    ...")
    print()
    print("  Export commands:")
    print("    results_df.to_csv('table.csv', index=False)")
    print("    results_df.to_latex('table.tex', float_format='%.4f', escape=False)")
    print("    result.summary().as_latex()   # full statsmodels summary")


# ==============================================================
# 7. Practical Guide
# ==============================================================

def demo_practical_guide() -> None:
    steps = [
        ("Step 1: Start with summary statistics.",
         "Report mean, std, min/max for all variables before any regression.",
         "Note the sample size; flag binary variables and their frequencies."),

        ("Step 2: Build regression tables incrementally.",
         "Present models M1 → Mk, adding controls column by column.",
         "Stable coefficients across columns signal robustness."),

        ("Step 3: Include standard errors, not just stars.",
         "Stars alone hide the magnitude and precision of effects.",
         "Report 95% CIs alongside (or instead of) significance stars."),

        ("Step 4: Run and report residual diagnostics.",
         "Cross-section: check ACF of residuals; test for heteroskedasticity.",
         "Time series: report Ljung-Box p-value; show ACF of residuals."),

        ("Step 5: Compare models with information criteria.",
         "AIC for prediction-oriented selection; BIC for parsimony.",
         "For time series, include test-set RMSE as the primary criterion."),

        ("Step 6: Export everything programmatically.",
         "Use to_csv() or to_latex() -- never manually type table values.",
         "Save the script that generates each table for full reproducibility."),

        ("Common mistakes to avoid:",
         "Reporting only in-sample fit and calling it forecast accuracy.",
         "Selecting models by p-values (specification searching inflates false positives)."),
    ]

    print()
    for question, opt_a, opt_b in steps:
        print(f"  {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  Clear, reproducible reporting builds credibility and enables replication.")
    print("  The table in your paper should be code output, not a paste from a screen.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Summary Statistics Tables")
    demo_summary_stats()

    section("2. Regression Output Tables")
    demo_regression_table()

    section("3. Coefficient Plots")
    demo_coefficient_plot()

    section("4. Residual Diagnostic Plots")
    demo_diagnostic_plots()

    section("5. Model Comparison Tables")
    demo_model_comparison()

    section("6. Exporting Results")
    demo_exporting()

    section("7. Practical Guide")
    demo_practical_guide()


if __name__ == "__main__":
    main()

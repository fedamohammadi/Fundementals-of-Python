"""
Linear Regression and OLS with statsmodels:
- Simulating data from a known linear DGP
- OLS estimator for simple regression (closed-form derivation)
- OLS estimator for multiple regression (matrix formula)
- Fitting models with statsmodels and reading the output
- Fitted values, residuals, and the R² decomposition
- The difference between population parameters and OLS estimates

"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# 1. Simulating Data from a Known DGP
# ==============================================================
# A data generating process (DGP) specifies the true model.
# Working with a known DGP lets us verify that OLS actually
# recovers the parameters we planted in the simulation.
#
# We use a Mincer earnings equation (workhorse of labor economics):
#   wage_i = β₀ + β₁·educ_i + β₂·exper_i + ε_i
#
# True parameters (OLS must find these from the simulated data):
#   β₀ =  8.0   baseline hourly wage ($) with zero educ and exper
#   β₁ =  1.5   each extra year of schooling raises wage by $1.50
#   β₂ =  0.5   each extra year of experience raises wage by $0.50
#   ε  ~ N(0, 2²)   unobserved factors (ability, luck, etc.)

TRUE_INTERCEPT = 8.0
TRUE_EDUC      = 1.5
TRUE_EXPER     = 0.5
ERROR_STD      = 2.0


def make_wage_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Simulate n observations from the Mincer DGP above.

    educ  ~ Uniform integer [8, 21]  (years of schooling)
    exper ~ Uniform integer [0, 40]  (years of experience)
    educ and exper are drawn independently — this matters in Section 2.

    Returns a DataFrame with columns: wage, educ, exper.
    """
    rng = np.random.default_rng(seed)

    educ  = rng.integers(8, 22, size=n).astype(float)
    exper = rng.integers(0, 41, size=n).astype(float)
    error = rng.normal(0, ERROR_STD, size=n)

    wage = TRUE_INTERCEPT + TRUE_EDUC * educ + TRUE_EXPER * exper + error

    return pd.DataFrame({"wage": wage, "educ": educ, "exper": exper})


def demo_dgp() -> None:
    df = make_wage_data()

    print(f"\n  DGP: wage = {TRUE_INTERCEPT} + {TRUE_EDUC}*educ + {TRUE_EXPER}*exper + eps")
    print(f"       eps ~ N(0, {ERROR_STD}^2),   n = {len(df)}")
    print()
    print(f"  {'Variable':>10} | {'Mean':>10} | {'Std':>10} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for col in ["wage", "educ", "exper"]:
        print(f"  {col:>10} | {df[col].mean():>10.2f} | {df[col].std():>10.2f} | "
              f"{df[col].min():>8.1f} | {df[col].max():>8.1f}")


# ==============================================================
# 2. OLS Estimator — Simple Regression by Hand
# ==============================================================
# Ordinary Least Squares minimises the sum of squared residuals:
#   RSS(α, β) = Σᵢ (yᵢ - α - β·xᵢ)²
#
# Setting ∂RSS/∂β = 0 and ∂RSS/∂α = 0 and solving yields:
#   β̂ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²  =  Cov(x,y) / Var(x)
#   α̂ = ȳ - β̂·x̄
#
# Intuition: β̂ is the slope that best aligns the variation in x
# with the variation in y.  α̂ ensures the line passes through (x̄, ȳ).

def ols_simple(
    x: list[float],
    y: list[float],
) -> tuple[float, float]:
    """
    OLS slope and intercept for the model  y = α + β·x + ε.

    Returns (alpha_hat, beta_hat).
    """
    n     = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n

    cov_xy = sum((x[i] - x_bar) * (y[i] - y_bar) for i in range(n))
    var_x  = sum((x[i] - x_bar) ** 2              for i in range(n))

    beta_hat  = cov_xy / var_x
    alpha_hat = y_bar - beta_hat * x_bar

    return alpha_hat, beta_hat


def demo_ols_simple() -> None:
    """
    Regress wage on educ only (omitting exper).

    Because educ and exper are independent by construction, β̂₁ is
    still unbiased for the true β₁ = 1.5.  But the intercept is NOT
    8.0 — it absorbs the average contribution of the omitted variable:
        α̂  ≈  β₀ + β₂·E[exper]  =  8.0 + 0.5 × 20  =  18.0

    This is a simple instance of omitted variable bias on the intercept.
    When educ and exper are correlated, the slope would be biased too.
    """
    df = make_wage_data()

    alpha_hat, beta_hat = ols_simple(
        x=df["educ"].tolist(),
        y=df["wage"].tolist(),
    )

    predicted_alpha = TRUE_INTERCEPT + TRUE_EXPER * df["exper"].mean()

    print(f"\n  Model: wage = a + b*educ + eps   (exper omitted)")
    print()
    print(f"  {'Parameter':>16} | {'True value':>14} | {'OLS estimate':>14}")
    print(f"  {'-'*16}-+-{'-'*14}-+-{'-'*14}")
    print(f"  {'a (intercept)':>16} | {predicted_alpha:>14.4f} | {alpha_hat:>14.4f}")
    print(f"  {'b1 (educ)':>16} | {TRUE_EDUC:>14.4f} | {beta_hat:>14.4f}")
    print()
    print(f"  Expected a_hat ~= b0 + b2*E[exper] = {TRUE_INTERCEPT} + {TRUE_EXPER}*{df['exper'].mean():.1f} = {predicted_alpha:.1f}")
    print("  b1_hat ~= 1.5  (unbiased because educ and exper are independent in our DGP)")


# ==============================================================
# 3. OLS Estimator — Multiple Regression (Matrix Formula)
# ==============================================================
# With k regressors, stacking all observations into matrices is
# cleaner than writing k separate first-order conditions.
#
# Model in matrix form:
#   Y  = X β + ε
#   Y  : (n×1)     response vector
#   X  : (n×(k+1)) design matrix — first column is all 1s (intercept)
#   β  : ((k+1)×1) parameter vector
#   ε  : (n×1)     error vector
#
# OLS minimises ε'ε.  The solution satisfies the normal equations:
#   (X'X) β̂ = X'Y   →   β̂ = (X'X)⁻¹ X'Y
#
# We solve the normal equations directly instead of computing (X'X)⁻¹
# explicitly — np.linalg.solve is faster and more numerically stable.

def ols_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS estimator via the normal equations (X'X)β̂ = X'Y.

    Parameters
    ----------
    X : (n × p) design matrix, must already include a column of 1s
    y : (n,) response vector

    Returns
    -------
    beta_hat : (p,) array of OLS coefficients
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


def demo_ols_matrix() -> None:
    """
    Regress wage on educ and exper using the matrix formula.
    Both variables are now included, so we expect estimates near
    the true β₀=8, β₁=1.5, β₂=0.5 (differences are sampling noise).
    """
    df = make_wage_data()
    n  = len(df)

    # Build design matrix X = [1 | educ | exper]
    ones = np.ones(n)
    X = np.column_stack([ones, df["educ"].values, df["exper"].values])
    y = df["wage"].values

    beta_hat = ols_matrix(X, y)

    print(f"\n  Model: wage = b0 + b1*educ + b2*exper + eps")
    print()
    print(f"  {'Parameter':>16} | {'True value':>12} | {'OLS estimate':>14}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*14}")
    labels    = ["b0 (intercept)", "b1 (educ)", "b2 (exper)"]
    true_vals = [TRUE_INTERCEPT, TRUE_EDUC, TRUE_EXPER]
    for label, true, est in zip(labels, true_vals, beta_hat):
        print(f"  {label:>16} | {true:>12.4f} | {est:>14.4f}")

    print()
    print("  With exper included, the intercept is now close to 8.0.")
    print("  Remaining gaps are sampling error (would shrink with larger n).")


# ==============================================================
# 4. OLS with statsmodels
# ==============================================================
# statsmodels.formula.api accepts Patsy formula strings — the same
# syntax as R's lm().  It adds an intercept by default (use -1 to
# suppress it).
#
#   smf.ols("y ~ x1 + x2", data=df).fit()
#
# The resulting RegressionResults object holds:
#   .params       — β̂ for each term
#   .bse          — SE(β̂) for each term
#   .tvalues      — t-statistics
#   .pvalues      — two-sided p-values
#   .rsquared     — R²
#   .rsquared_adj — Adjusted R²
#   .fvalue       — F-statistic (tests all slopes = 0 jointly)
#   .fittedvalues — ŷ
#   .resid        — ê = y - ŷ
#   .summary()    — full printed table

def demo_statsmodels() -> None:
    df = make_wage_data()
    result = smf.ols("wage ~ educ + exper", data=df).fit()

    print(f"\n{result.summary()}")

    # Quick programmatic access for use in downstream calculations
    print("\n  Programmatic access to key attributes:")
    print(f"    result.params   :  {dict(result.params.round(4))}")
    print(f"    result.bse      :  {dict(result.bse.round(4))}")
    print(f"    result.rsquared :  {result.rsquared:.4f}")


# ==============================================================
# 5. Reading the Regression Table
# ==============================================================
# Every number in the statsmodels summary table has a precise meaning.
# We reconstruct the key statistics from first principles to show
# exactly what the table is reporting.
#
# For each coefficient β̂_j:
#   SE(β̂_j)  — standard deviation of β̂_j across hypothetical samples
#               (derived from the variance matrix Var(β̂) = σ²(X'X)⁻¹)
#   t_j      = β̂_j / SE(β̂_j)   (how many SEs away from zero)
#   p_j      = 2·P(T > |t_j|)   where T ~ t(n − k − 1)
#               (probability of seeing this t-stat if β_j = 0)

def demo_reading_output() -> None:
    df = make_wage_data()
    n  = len(df)
    k  = 2   # regressors: educ, exper (intercept is not counted)

    result    = smf.ols("wage ~ educ + exper", data=df).fit()
    df_resid  = n - k - 1   # degrees of freedom for residuals

    print(f"\n  n = {n},  k = {k},  df_residual = {df_resid}")
    print()
    print(f"  {'Variable':>14} | {'b_hat':>10} | {'SE(b_hat)':>9} | "
          f"{'t=b/SE':>10} | {'p-value':>9}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}")

    for name in result.params.index:
        beta   = result.params[name]
        se     = result.bse[name]
        t_stat = beta / se          # replicates result.tvalues[name]
        p_val  = result.pvalues[name]
        print(f"  {name:>14} | {beta:>10.4f} | {se:>9.4f} | "
              f"{t_stat:>10.4f} | {p_val:>9.4f}")

    print()
    print(f"  R^2     = {result.rsquared:.4f}  ->  model explains "
          f"{result.rsquared*100:.1f}% of wage variation")
    print(f"  Adj R^2 = {result.rsquared_adj:.4f}  (penalises adding irrelevant regressors)")
    print()
    print(f"  F({k}, {df_resid}) = {result.fvalue:.2f},  p = {result.f_pvalue:.2e}")
    print("  The F-test checks H0: b1 = b2 = 0 simultaneously.")
    print("  p ~= 0 means at least one regressor has explanatory power.")
    print()
    print("  Rule of thumb for t-stat significance (two-sided, alpha=0.05):")
    print("  |t| > ~2.0 when df is large (t-distribution -> normal).")
    for name in result.params.index:
        t = result.tvalues[name]
        sig = "significant at 5%" if abs(t) > 2.0 else "NOT significant"
        print(f"    {name:>14}: |t| = {abs(t):.2f}  ->  {sig}")


# ==============================================================
# 6. Fitted Values, Residuals, and R²
# ==============================================================
# OLS partitions each observation into a fitted value and a residual:
#   yᵢ = ŷᵢ + êᵢ
#   ŷᵢ = β̂₀ + β̂₁·x₁ᵢ + β̂₂·x₂ᵢ   (the model's prediction for unit i)
#   êᵢ = yᵢ − ŷᵢ                   (what the model got wrong)
#
# Two algebraic properties hold for any OLS fit (not assumptions):
#   Σ êᵢ = 0                        residuals sum to zero
#   Σ xⱼᵢ êᵢ = 0  for all j        residuals are orthogonal to X
#
# These lead to the ANOVA decomposition:
#   SS_tot = SS_reg + SS_res
#   Σ(yᵢ − ȳ)² = Σ(ŷᵢ − ȳ)² + Σêᵢ²
#
#   R² = SS_reg / SS_tot  =  1 − SS_res / SS_tot  ∈ [0, 1]

def demo_r_squared() -> None:
    """
    Compute R² manually from the residuals and verify it matches
    statsmodels.  Also confirm the two OLS residual properties.
    """
    df     = make_wage_data()
    result = smf.ols("wage ~ educ + exper", data=df).fit()

    y     = df["wage"].values
    y_hat = result.fittedvalues.values
    e_hat = result.resid.values        # ê  (not the true ε)

    y_bar  = y.mean()
    ss_tot = np.sum((y - y_bar) ** 2)
    ss_res = np.sum(e_hat ** 2)
    ss_reg = np.sum((y_hat - y_bar) ** 2)
    r2     = ss_reg / ss_tot           # equivalent to 1 - ss_res / ss_tot

    print(f"\n  SS_tot = {ss_tot:>12,.2f}  (total variation in wage)")
    print(f"  SS_reg = {ss_reg:>12,.2f}  (explained by educ + exper)")
    print(f"  SS_res = {ss_res:>12,.2f}  (unexplained residual)")
    print()
    print(f"  SS_reg + SS_res = {ss_reg + ss_res:>12,.2f}  (must equal SS_tot)")
    print()
    print(f"  R^2 (manual)      = {r2:.6f}")
    print(f"  R^2 (statsmodels) = {result.rsquared:.6f}")
    print()

    # These are ~= 0 by construction — a useful sanity check after any OLS fit
    print("  OLS residual properties (~= 0 by construction, not by luck):")
    print(f"    sum(e_i)          = {np.sum(e_hat):>+.2e}")
    print(f"    sum(educ * e_i)   = {np.sum(df['educ'].values  * e_hat):>+.2e}")
    print(f"    sum(exper * e_i)  = {np.sum(df['exper'].values * e_hat):>+.2e}")
    print()
    print("  If any of these were large, it would indicate a programming error.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. Data Generating Process (Mincer Earnings Equation)")
    demo_dgp()

    section("2. OLS by Hand - Simple Regression (Closed Form)")
    demo_ols_simple()

    section("3. OLS by Hand - Multiple Regression (Matrix Form)")
    demo_ols_matrix()

    section("4. OLS with statsmodels")
    demo_statsmodels()

    section("5. Reading the Regression Table")
    demo_reading_output()

    section("6. Fitted Values, Residuals, and R^2")
    demo_r_squared()


if __name__ == "__main__":
    main()

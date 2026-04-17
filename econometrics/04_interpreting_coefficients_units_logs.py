"""
Interpreting OLS Coefficients - Units, Logs, and Elasticities:
- Why functional form determines the meaning of a coefficient
- Level-level model: direct unit interpretation
- Log-level model: semi-elasticity (% change in y per unit change in x)
- Level-log model: semi-elasticity (unit change in y per % change in x)
- Log-log model: elasticity (% change in y per % change in x)
- Standardized coefficients: effect measured in standard deviation units

The same data, four different models — the coefficient changes meaning
entirely depending on whether y and x enter in levels or logs.
"""

import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# We simulate a labour-market dataset with three variables:
#   wage  : hourly wage in USD (positive, right-skewed in reality)
#   educ  : years of schooling (8-21)
#   exper : years of work experience (0-40)
#
# True DGP uses a log-level structure for wage (realistic):
#   log(wage) = 1.6 + 0.10*educ + 0.008*exper + eps
#
# Exponentiating: wage = exp(1.6 + 0.10*educ + 0.008*exper + eps)
# This produces the right-skewed wage distribution seen in real data.

TRUE_LOG_INTERCEPT = 1.6
TRUE_LOG_EDUC      = 0.10    # 10% wage increase per year of schooling
TRUE_LOG_EXPER     = 0.008   # 0.8% wage increase per year of experience
ERROR_STD          = 0.25    # log-scale noise


def make_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate n observations from the log-level DGP above.

    Returns columns: wage, log_wage, educ, exper, log_educ, log_exper.
    Pre-computing logs here avoids repeating math.log() in every demo.
    """
    rng = np.random.default_rng(seed)

    educ  = rng.integers(8, 22, size=n).astype(float)
    exper = rng.integers(0, 41, size=n).astype(float)
    eps   = rng.normal(0, ERROR_STD, size=n)

    log_wage = TRUE_LOG_INTERCEPT + TRUE_LOG_EDUC * educ + TRUE_LOG_EXPER * exper + eps
    wage     = np.exp(log_wage)

    return pd.DataFrame({
        "wage":     wage,
        "log_wage": log_wage,
        "educ":     educ,
        "exper":    exper,
        "log_educ": np.log(educ),
        # Suppress the divide-by-zero warning from log(0); the np.where already
        # replaces those values with NaN, but numpy evaluates both branches first.
        "log_exper": np.where(exper > 0, np.log(np.maximum(exper, 1e-9)), np.nan),
    })


# ==============================================================
# 1. Why Functional Form Matters
# ==============================================================
# The coefficient b1 in a regression means something different
# depending on whether y and x enter in levels or logs.
#
#   Model          |  Equation             | Interpretation of b1
#   ---------------+-----------------------+-------------------------------------
#   Level - Level  |  y     = b0 + b1*x   | +1 unit x  ->  +b1 units y
#   Log   - Level  |  ln(y) = b0 + b1*x   | +1 unit x  ->  +b1*100% in y (approx)
#   Level - Log    |  y     = b0 + b1*ln(x)| +1% in x  ->  +b1/100 units y
#   Log   - Log    |  ln(y) = b0 + b1*ln(x)| +1% in x  ->  +b1% in y (elasticity)
#
# Choosing the wrong functional form does not just affect fit —
# it changes what economic question you are answering.

def demo_why_form_matters() -> None:
    """
    Show the same educ coefficient across all four model types
    on identical data.  The number changes dramatically, and each
    number answers a different question.
    """
    df = make_data()

    # Fit all four model types using the same two regressors
    m_ll  = smf.ols("wage     ~ educ + exper",             data=df).fit()
    m_lg  = smf.ols("log_wage ~ educ + exper",             data=df).fit()
    m_lv  = smf.ols("wage     ~ log_educ + exper",         data=df).fit()
    m_gg  = smf.ols("log_wage ~ log_educ + exper",         data=df).fit()

    b_ll  = m_ll.params["educ"]
    b_lg  = m_lg.params["educ"]
    b_lv  = m_lv.params["log_educ"]
    b_gg  = m_gg.params["log_educ"]

    print(f"\n  {'Model':>12} | {'educ coeff':>12} | What it means")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*45}")
    print(f"  {'Level-Level':>12} | {b_ll:>12.4f} | +1 yr educ -> +${b_ll:.2f}/hr wage")
    print(f"  {'Log-Level':>12} | {b_lg:>12.4f} | +1 yr educ -> +{b_lg*100:.1f}% wage (approx)")
    print(f"  {'Level-Log':>12} | {b_lv:>12.4f} | +1% educ -> +${b_lv/100:.3f}/hr wage")
    print(f"  {'Log-Log':>12} | {b_gg:>12.4f} | +1% educ -> +{b_gg:.2f}% wage (elasticity)")
    print()
    print("  Same data. Same question. Four different numbers.")
    print("  The coefficient is meaningless without knowing the functional form.")


# ==============================================================
# 2. Level-Level Model
# ==============================================================
# The most literal interpretation: coefficients are in the original
# units of y and x.
#
#   wage = b0 + b1*educ + b2*exper + eps
#
# Interpretation:
#   b1 = dE[wage]/d(educ)
#      = the expected dollar-per-hour increase in wage for one
#        additional year of schooling, holding exper fixed.
#
# Limitation: this model imposes a linear wage profile — the
# 8th year of school raises wages by the same amount as the 18th.
# Log models relax this by modelling proportional effects.

def demo_level_level() -> None:
    """
    Fit level-level and walk through coefficient interpretation
    with a concrete prediction example to make the math tangible.
    """
    df = make_data()
    result = smf.ols("wage ~ educ + exper", data=df).fit()

    b0, b1, b2 = result.params["Intercept"], result.params["educ"], result.params["exper"]

    print(f"\n  Fitted model:  wage = {b0:.3f} + {b1:.3f}*educ + {b2:.3f}*exper")
    print()
    print("  Coefficient interpretation:")
    print(f"    b1 = {b1:.3f}  ->  each extra year of schooling adds ${b1:.2f}/hr to wage")
    print(f"    b2 = {b2:.3f}  ->  each extra year of experience adds ${b2:.2f}/hr to wage")
    print(f"    b0 = {b0:.3f}  ->  predicted wage with educ=0, exper=0 (extrapolation)")
    print()

    # Concrete prediction: compare two workers
    w_base = b0 + b1 * 12 + b2 * 5
    w_more = b0 + b1 * 16 + b2 * 5
    diff   = w_more - w_base
    print("  Prediction example (exper=5 fixed):")
    print(f"    educ=12 (high-school):  predicted wage = ${w_base:.2f}/hr")
    print(f"    educ=16 (college):      predicted wage = ${w_more:.2f}/hr")
    print(f"    Difference (4 yrs):     ${diff:.2f}/hr  =  4 * b1 = 4 * {b1:.3f}")
    print()
    print(f"  R^2 = {result.rsquared:.4f}   (the log-level model fits better - see Section 3)")


# ==============================================================
# 3. Log-Level Model
# ==============================================================
# The most common model in labour economics:
#   ln(wage) = b0 + b1*educ + b2*exper + eps
#
# Differentiating with respect to educ:
#   d(ln wage)/d(educ) = b1
#   (1/wage) * d(wage)/d(educ) = b1
#   d(wage)/wage = b1 * d(educ)
#
# So b1 is the proportional change in wage per unit change in educ.
# In percentage terms: a 1-unit increase in educ changes wage by
# approximately b1*100%.
#
# The approximation is exact for infinitesimal changes.  For
# discrete changes, the exact formula is:
#   % change in wage = (exp(b1) - 1) * 100
#
# Rule of thumb: for |b1| < 0.10 the approximation is fine;
# for larger coefficients always report exp(b1) - 1.

def demo_log_level() -> None:
    """
    Fit ln(wage) ~ educ + exper, interpret the educ coefficient
    both via the approximation and the exact exponential formula,
    and compare R^2 with the level-level model from Section 2.
    """
    df = make_data()
    result = smf.ols("log_wage ~ educ + exper", data=df).fit()

    b0 = result.params["Intercept"]
    b1 = result.params["educ"]
    b2 = result.params["exper"]

    # Approximate interpretation (valid for small b)
    pct_approx_educ  = b1 * 100
    pct_approx_exper = b2 * 100

    # Exact interpretation using exp(b) - 1
    pct_exact_educ  = (math.exp(b1) - 1) * 100
    pct_exact_exper = (math.exp(b2) - 1) * 100

    print(f"\n  Fitted model:  ln(wage) = {b0:.3f} + {b1:.4f}*educ + {b2:.4f}*exper")
    print()
    print("  Educ coefficient:")
    print(f"    b1 = {b1:.4f}")
    print(f"    Approx:  b1 * 100   = {pct_approx_educ:.2f}% wage increase per extra year of school")
    print(f"    Exact:   (exp(b1)-1)*100 = {pct_exact_educ:.2f}%  (use this when b1 > 0.10)")
    print()
    print("  Exper coefficient:")
    print(f"    b2 = {b2:.4f}")
    print(f"    Approx:  {pct_approx_exper:.2f}% wage increase per extra year of experience")
    print(f"    Exact:   {pct_exact_exper:.2f}%")
    print()

    # Concrete prediction using the log-level model
    lw_base = b0 + b1 * 12 + b2 * 5
    lw_more = b0 + b1 * 16 + b2 * 5
    w_base  = math.exp(lw_base)
    w_more  = math.exp(lw_more)

    print("  Prediction example (exper=5 fixed):")
    print(f"    educ=12:  ln(wage) = {lw_base:.3f}  ->  wage = exp({lw_base:.3f}) = ${w_base:.2f}/hr")
    print(f"    educ=16:  ln(wage) = {lw_more:.3f}  ->  wage = exp({lw_more:.3f}) = ${w_more:.2f}/hr")
    print(f"    4-yr effect: +{(w_more/w_base - 1)*100:.1f}%  (4 * b1*100 = {4*pct_approx_educ:.1f}% approx)")
    print()
    print(f"  R^2 = {result.rsquared:.4f}   (better fit than level-level {0.6674:.4f})")
    print("  This makes sense: the DGP was log-level, so the model is correctly specified.")


# ==============================================================
# 4. Level-Log Model
# ==============================================================
# Less common, but useful when the REGRESSOR has diminishing returns:
#   wage = b0 + b1*ln(educ) + b2*exper + eps
#
# Differentiating with respect to educ:
#   d(wage)/d(educ) = b1 * (1/educ)
#
# Rearranging in terms of a 1% change in educ (d(educ)/educ = 0.01):
#   d(wage) = b1 * (d(educ)/educ) = b1 * 0.01
#
# So: a 1% increase in educ raises wage by b1/100 dollars.
# Equivalently: doubling educ (100% increase) raises wage by b1 dollars.
#
# This model says the RETURN to schooling is diminishing:
# going from 8 to 9 years of school has a larger absolute effect
# than going from 18 to 19 years.

def demo_level_log() -> None:
    """
    Fit wage ~ log(educ) + exper and interpret the log-educ coefficient.
    Show how to answer "what is the predicted wage change for a 10% and
    100% increase in educ?"
    """
    df = make_data()
    result = smf.ols("wage ~ log_educ + exper", data=df).fit()

    b0 = result.params["Intercept"]
    b1 = result.params["log_educ"]   # coefficient on ln(educ)
    b2 = result.params["exper"]

    print(f"\n  Fitted model:  wage = {b0:.3f} + {b1:.3f}*ln(educ) + {b2:.3f}*exper")
    print()
    print("  Coefficient on ln(educ):")
    print(f"    b1 = {b1:.3f}")
    print(f"    +1% in educ  -> +b1/100 = +${b1/100:.3f}/hr")
    print(f"    +10% in educ -> +b1*ln(1.10) = +${b1 * math.log(1.10):.3f}/hr")
    print(f"    +100% in educ (doubling) -> +b1*ln(2) = +${b1 * math.log(2):.3f}/hr")
    print()

    # Diminishing returns: compare marginal effect at low vs high educ
    # d(wage)/d(educ) = b1 / educ
    marg_8  = b1 / 8
    marg_16 = b1 / 16
    print("  Diminishing marginal returns (d(wage)/d(educ) = b1/educ):")
    print(f"    At educ=8:   one more year of school -> +${marg_8:.3f}/hr")
    print(f"    At educ=16:  one more year of school -> +${marg_16:.3f}/hr")
    print(f"    The return at educ=8 is {marg_8/marg_16:.1f}x larger than at educ=16.")
    print()

    # Prediction comparison
    w_base = b0 + b1 * math.log(12) + b2 * 5
    w_more = b0 + b1 * math.log(16) + b2 * 5
    print("  Prediction example (exper=5 fixed):")
    print(f"    educ=12:  wage = {b0:.3f} + {b1:.3f}*ln(12) + {b2:.3f}*5 = ${w_base:.2f}/hr")
    print(f"    educ=16:  wage = {b0:.3f} + {b1:.3f}*ln(16) + {b2:.3f}*5 = ${w_more:.2f}/hr")
    print(f"    Difference: ${w_more - w_base:.2f}/hr  =  b1 * ln(16/12) = {b1:.3f} * {math.log(16/12):.4f}")
    print()
    print(f"  R^2 = {result.rsquared:.4f}")


# ==============================================================
# 5. Log-Log Model (Elasticity)
# ==============================================================
# The classic elasticity specification:
#   ln(y) = b0 + b1*ln(x) + controls + eps
#
# Differentiating:
#   d(ln y)/d(ln x) = b1
#   (dy/y) / (dx/x) = b1
#
# So b1 is the elasticity of y with respect to x:
#   a 1% increase in x leads to a b1% increase in y.
#
# This is scale-free — it does not matter whether wages are in
# dollars or euros.  Elasticities are directly comparable across
# different studies and countries.
#
# Common applications:
#   - Price elasticity of demand: ln(Q) ~ ln(P)
#   - Income elasticity of demand: ln(Q) ~ ln(income)
#   - Wage elasticity: ln(wage) ~ ln(educ)

def demo_log_log() -> None:
    """
    Fit ln(wage) ~ ln(educ) + exper and interpret the log-log
    coefficient as an elasticity.  Also show the symmetry property:
    the educ elasticity is the same whether educ is high or low.
    """
    df = make_data()

    # Note: exper enters in levels here because a "% change in exper"
    # is hard to interpret (going from 1 to 2 years = 100% increase).
    result = smf.ols("log_wage ~ log_educ + exper", data=df).fit()

    b0 = result.params["Intercept"]
    b1 = result.params["log_educ"]    # elasticity of wage w.r.t. educ
    b2 = result.params["exper"]

    print(f"\n  Fitted model:  ln(wage) = {b0:.3f} + {b1:.4f}*ln(educ) + {b2:.4f}*exper")
    print()
    print(f"  b1 = {b1:.4f}  ->  wage elasticity with respect to educ")
    print(f"    +1% in educ  ->  +{b1:.2f}% in wage")
    print(f"    +10% in educ ->  +{b1 * math.log(1.10) / math.log(1.10) * 10:.2f}% in wage  (exact: {(1.10**b1 - 1)*100:.2f}%)")
    print(f"    +100% in educ (doubling) -> +{(2**b1 - 1)*100:.1f}% in wage")
    print()

    # Symmetry: the elasticity is the same at all educ levels
    # (unlike the level-log model where the marginal effect changes)
    print("  Elasticity is constant (does not depend on the value of educ):")
    for educ_val in [8, 12, 16, 20]:
        # d(wage)/d(educ) in level terms = b1 * (wage/educ)
        # But the % change interpretation is always b1% per 1% change
        print(f"    At educ={educ_val}: a 1% increase in educ -> {b1:.2f}% increase in wage")
    print()

    # Compare with the true elasticity implied by the DGP
    # True DGP: ln(wage) = 1.6 + 0.10*educ + 0.008*exper
    # For the log-log model to match, the elasticity at the mean educ
    # should approximate: d(ln wage)/d(ln educ) = 0.10 * mean_educ
    true_elast_at_mean = TRUE_LOG_EDUC * df["educ"].mean()
    print(f"  True DGP is log-level, not log-log.")
    print(f"  Local elasticity implied by DGP at mean educ ({df['educ'].mean():.1f} yrs):")
    print(f"    b1_dgp * E[educ] = {TRUE_LOG_EDUC} * {df['educ'].mean():.1f} = {true_elast_at_mean:.3f}")
    print(f"  OLS log-log estimate: {b1:.4f}  (close, because log-log approximates log-level locally)")
    print()
    print(f"  R^2 = {result.rsquared:.4f}")


# ==============================================================
# 6. Standardized Coefficients
# ==============================================================
# Problem: when regressors are on different scales (years of school
# vs. years of experience), the raw coefficients are not comparable.
# b1=2.74 ($/yr of school) vs b2=0.18 ($/yr of experience) does NOT
# mean schooling matters 15x more — the two are in different units.
#
# Solution: standardize each variable to mean=0, std=1 before regressing.
#   z_x = (x - mean(x)) / std(x)
#
# The standardized coefficient ("beta weight") answers:
#   "A 1-standard-deviation increase in x changes y by how many
#    standard deviations (or units, if y is not standardized)?"
#
# This makes educ and exper directly comparable.

def standardize(series: pd.Series) -> pd.Series:
    """Return the z-score of a Series: (x - mean) / std."""
    return (series - series.mean()) / series.std()


def demo_standardized() -> None:
    """
    Fit the level-level model twice — once raw, once with standardized
    regressors — and compare the coefficient magnitudes directly.
    """
    df = make_data()

    # Raw coefficients (not comparable across variables)
    raw = smf.ols("wage ~ educ + exper", data=df).fit()

    # Standardize only the regressors (leave wage in dollar units so
    # coefficients = $ change in wage per 1 SD change in regressor)
    df["educ_z"]  = standardize(df["educ"])
    df["exper_z"] = standardize(df["exper"])
    std = smf.ols("wage ~ educ_z + exper_z", data=df).fit()

    educ_sd  = df["educ"].std()
    exper_sd = df["exper"].std()

    print(f"\n  Variable standard deviations:")
    print(f"    SD(educ)  = {educ_sd:.2f} years")
    print(f"    SD(exper) = {exper_sd:.2f} years")
    print()
    print(f"  {'':>10} | {'Raw coeff':>12} | {'Std coeff':>12} | {'Std coeff interpretation'}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*40}")

    for var, var_z, label in [("educ", "educ_z", "educ"), ("exper", "exper_z", "exper")]:
        b_raw = raw.params[var]
        b_std = std.params[var_z]
        sd    = df[var].std()
        print(f"  {label:>10} | {b_raw:>12.4f} | {b_std:>12.4f} | "
              f"+1 SD in {label} ({sd:.1f} yrs) -> +${b_std:.2f}/hr wage")

    print()
    b_educ_std  = std.params["educ_z"]
    b_exper_std = std.params["exper_z"]
    print("  Comparison: which matters more for wages?")
    print(f"    educ:  +1 SD  -> +${b_educ_std:.2f}/hr")
    print(f"    exper: +1 SD  -> +${b_exper_std:.2f}/hr")
    ratio = abs(b_educ_std / b_exper_std)
    print(f"    On a per-SD basis, educ has {ratio:.1f}x the wage effect of exper.")
    print()
    print("  Note: raw coefficients (2.74 vs 0.18) make educ look 15x more")
    print(f"  important, but that is partly because 1 yr of educ and 1 yr of exper")
    print(f"  are not equally 'large' moves ({educ_sd:.1f} vs {exper_sd:.1f} SD units).")

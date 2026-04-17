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

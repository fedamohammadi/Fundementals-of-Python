"""
Instrumental Variables and 2SLS:
- The endogeneity problem: why OLS is biased when X is correlated with the error
- Instrumental variable conditions: relevance and the exclusion restriction
- The Wald estimator: IV in the simplest case
- Two-stage least squares (2SLS) with one or multiple instruments
- First-stage diagnostics: testing for weak instruments
- The Hausman endogeneity test: when is IV actually needed?
- Overidentification: the Sargan-Hansen J-test
- A practical guide for IV estimation
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Classic returns-to-education setup (Card 1995 style).
# Workers decide how many years of schooling to get.
# Ability raises both wages AND schooling (the endogeneity source).
# Two instruments:
#   z1: grew up near a 4-year college (distance instrument)
#   z2: has an older sibling who attended college (peer instrument)
# Both shift education but have no direct effect on wages.
#
# True DGP: log_wage = 1.5 + 0.10 * educ + 0.5 * ability + eps

N_OBS        = 1200
TRUE_B_EDUC  = 0.10
TRUE_B_ABLT  = 0.50


def make_iv_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ability      = rng.normal(0, 1, N_OBS)
    near_college = rng.binomial(1, 0.45, N_OBS).astype(float)
    sibling_col  = rng.binomial(1, 0.35, N_OBS).astype(float)

    # Education endogenously determined by ability + instruments + noise
    educ_star = (12
                 + 0.8 * ability
                 + 1.0 * near_college
                 + 0.6 * sibling_col
                 + rng.normal(0, 1.5, N_OBS))
    educ = np.clip(educ_star, 8, 20).round()

    eps      = rng.normal(0, 0.20, N_OBS)
    log_wage = 1.5 + TRUE_B_EDUC * educ + TRUE_B_ABLT * ability + eps

    return pd.DataFrame({
        "log_wage":     log_wage,
        "educ":         educ,
        "ability":      ability,
        "near_college": near_college,
        "sibling_col":  sibling_col,
    })


# ==============================================================
# 1. The Endogeneity Problem
# ==============================================================
# OLS is unbiased when Cov(X, eps) = 0.  Here ability enters both
# the wage equation and the education decision, so it ends up in
# the wage residual for any researcher who cannot observe it.
#
# OVB formula:
#   plim(b_OLS) = b_true + Cov(educ, ability) / Var(educ) * b_ability
#
# Cov(educ, ability) > 0 and b_ability > 0, so OLS overstates b_educ.

def demo_endogeneity_problem() -> None:
    df = make_iv_data()

    oracle = smf.ols("log_wage ~ educ + ability", data=df).fit()
    naive  = smf.ols("log_wage ~ educ",           data=df).fit()

    corr = df[["educ", "ability"]].corr().loc["educ", "ability"]

    print(f"\n  True b_educ = {TRUE_B_EDUC:.3f}")
    print(f"  Corr(educ, ability) = {corr:.3f}  (ability raises both wages and schooling)")
    print()
    print(f"  {'Estimator':>22} | {'b_educ':>8} | {'SE':>7} | Bias")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*7}-+-{'-'*15}")

    for label, res in [("Naive OLS (ability omitted)", naive),
                       ("Oracle OLS (ability observed)", oracle)]:
        b = res.params["educ"]
        print(f"  {label:>22} | {b:>8.4f} | {res.bse['educ']:>7.4f} | {b - TRUE_B_EDUC:+.4f}")

    print()
    print("  Naive OLS overstates the return to education because it picks up")
    print("  ability's wage effect.  IV breaks this by finding variation in educ")
    print("  that comes only from the instrument, not from ability.")


# ==============================================================
# 2. Instrumental Variable Conditions
# ==============================================================
# A valid instrument Z must satisfy two conditions:
#
#   1. Relevance: Cov(Z, X) != 0.  Z must genuinely affect the
#      endogenous variable.  Weak instruments (small first-stage F)
#      inflate IV SEs and bias the estimator toward OLS in finite samples.
#
#   2. Exclusion restriction: Cov(Z, eps) = 0.  Z affects Y only
#      through X.  This is an untestable assumption defended by
#      economic reasoning, not statistics.
#
# near_college and sibling_col:
#   Relevance: college proximity and peer effects shift schooling (testable).
#   Exclusion: proximity / sibling attendance don't directly raise wages
#              except by inducing more education.

def demo_iv_conditions() -> None:
    df = make_iv_data()

    print(f"\n  Relevance: first-stage regression of educ on each instrument")
    print()
    print(f"  {'Instrument':>14} | {'b':>7} | {'SE':>6} | {'F':>7} | Verdict")
    print(f"  {'-'*14}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*15}")

    for z in ["near_college", "sibling_col"]:
        res = smf.ols(f"educ ~ {z}", data=df).fit()
        b   = res.params[z]
        se  = res.bse[z]
        f   = (b / se) ** 2
        verdict = "strong" if f > 10 else "weak -- concern"
        print(f"  {z:>14} | {b:>7.3f} | {se:>6.3f} | {f:>7.1f} | {verdict}")

    print()
    print("  Exclusion restriction (untestable by definition):")
    print("    near_college: wages don't depend on where you grew up once")
    print("                  education is controlled for.")
    print("    sibling_col:  sibling's college attendance is not a direct")
    print("                  wage signal to employers.")
    print()

    print("  Reduced-form check (Z -> log_wage, no X control):")
    for z in ["near_college", "sibling_col"]:
        rf = smf.ols(f"log_wage ~ {z}", data=df).fit()
        print(f"    {z:<14}: b = {rf.params[z]:.4f}   p = {rf.pvalues[z]:.4f}")
    print()
    print("  A significant reduced form confirms that Z shifts Y via X --")
    print("  necessary but not sufficient for validity (exclusion still needed).")
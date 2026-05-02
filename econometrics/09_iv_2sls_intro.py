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


# ==============================================================
# 3. The Wald Estimator
# ==============================================================
# With a single binary instrument Z, the IV estimator simplifies to:
#
#   b_IV = (ȳ_{Z=1} - ȳ_{Z=0}) / (x̄_{Z=1} - x̄_{Z=0})
#        = Reduced Form / First Stage
#
# This ratio says: for every extra year of education caused by Z,
# how much does log_wage change?  The denominator is the "first
# stage" effect and the numerator is the "reduced form" effect.
#
# The Wald estimator is the Local Average Treatment Effect (LATE):
# it identifies the return to education only for "compliers" --
# workers whose education changed because of the instrument.

def wald_estimator(df: pd.DataFrame, y: str, x: str, z: str) -> float:
    y1, y0 = df.loc[df[z] == 1, y].mean(), df.loc[df[z] == 0, y].mean()
    x1, x0 = df.loc[df[z] == 1, x].mean(), df.loc[df[z] == 0, x].mean()
    return (y1 - y0) / (x1 - x0)


def demo_wald_estimator() -> None:
    df = make_iv_data()

    print(f"\n  True b_educ = {TRUE_B_EDUC:.3f}")
    print()
    print(f"  {'Instrument':>14} | {'Reduced Form':>13} | {'First Stage':>12} | Wald IV")
    print(f"  {'-'*14}-+-{'-'*13}-+-{'-'*12}-+-{'-'*9}")

    for z in ["near_college", "sibling_col"]:
        rf  = df.loc[df[z] == 1, "log_wage"].mean() - df.loc[df[z] == 0, "log_wage"].mean()
        fs  = df.loc[df[z] == 1, "educ"].mean()     - df.loc[df[z] == 0, "educ"].mean()
        w   = wald_estimator(df, "log_wage", "educ", z)
        print(f"  {z:>14} | {rf:>13.4f} | {fs:>12.4f} | {w:.4f}")

    print()
    naive_b = smf.ols("log_wage ~ educ", data=df).fit().params["educ"]
    print(f"  Naive OLS b_educ = {naive_b:.4f}  (for comparison)")
    print()
    print("  Wald IV is close to the true 0.10, while OLS is ~50% higher.")
    print("  Each instrument gives a slightly different estimate because each")
    print("  identifies the LATE for its own subpopulation of compliers.")


# ==============================================================
# 4. Two-Stage Least Squares (2SLS)
# ==============================================================
# 2SLS is the standard IV estimator with one or more instruments:
#
#   Stage 1: regress X on Z (and any included controls) to get X̂.
#   Stage 2: regress Y on X̂ (and controls) to get b_IV.
#
# The key insight: X̂ contains only the part of X that's driven by
# Z -- by construction uncorrelated with ability in the error.
#
# With multiple instruments, 2SLS is more efficient than using each
# instrument separately (it combines their variation optimally).
#
# In statsmodels: use IV2SLS from linearmodels or implement manually.
# Here we implement it directly to show what happens under the hood.

def fit_2sls(df: pd.DataFrame, y: str, x_endog: str,
             instruments: list, controls: list = None) -> dict:
    """
    Manual 2SLS: returns a dict with b, se, t, p for each variable.
    controls are included in both stages (exogenous regressors).
    """
    controls = controls or []

    # Stage 1
    rhs1 = " + ".join(instruments + controls)
    stage1 = smf.ols(f"{x_endog} ~ {rhs1}", data=df).fit()
    df = df.copy()
    df[f"{x_endog}_hat"] = stage1.fittedvalues

    # Stage 2
    rhs2 = f"{x_endog}_hat" + (" + " + " + ".join(controls) if controls else "")
    stage2 = smf.ols(f"{y} ~ {rhs2}", data=df).fit()

    # Correct the residuals: stage 2 uses X_hat, so residuals use X (not X_hat)
    n  = len(df)
    k  = stage2.df_model + 1
    e  = df[y].values - stage2.params["Intercept"] - stage2.params[f"{x_endog}_hat"] * df[x_endog].values
    s2 = (e @ e) / (n - k)
    X2 = sm.add_constant(df[f"{x_endog}_hat"].values)
    vcov = s2 * np.linalg.inv(X2.T @ X2)
    se   = np.sqrt(np.diag(vcov))

    b_iv = stage2.params[f"{x_endog}_hat"]
    se_iv = se[1]
    t_iv  = b_iv / se_iv
    p_iv  = 2 * (1 - stats.t.cdf(abs(t_iv), df=n - k))

    return {"b": b_iv, "se": se_iv, "t": t_iv, "p": p_iv,
            "stage1_f": stage1.fvalue, "stage1_r2": stage1.rsquared}


def demo_2sls() -> None:
    df = make_iv_data()

    ols = smf.ols("log_wage ~ educ", data=df).fit()

    iv1 = fit_2sls(df, "log_wage", "educ", instruments=["near_college"])
    iv2 = fit_2sls(df, "log_wage", "educ", instruments=["sibling_col"])
    iv_both = fit_2sls(df, "log_wage", "educ",
                       instruments=["near_college", "sibling_col"])

    print(f"\n  True b_educ = {TRUE_B_EDUC:.3f}  (N = {N_OBS})")
    print()
    print(f"  {'Estimator':>22} | {'b_educ':>8} | {'SE':>7} | {'p':>8} | First-stage F")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*14}")

    print(f"  {'OLS':>22} | {ols.params['educ']:>8.4f} | {ols.bse['educ']:>7.4f} | "
          f"{ols.pvalues['educ']:>8.4f} | n/a")
    for label, res in [("IV (near_college)", iv1), ("IV (sibling_col)", iv2),
                       ("2SLS (both Z)",    iv_both)]:
        print(f"  {label:>22} | {res['b']:>8.4f} | {res['se']:>7.4f} | "
              f"{res['p']:>8.4f} | {res['stage1_f']:>14.1f}")

    print()
    print("  2SLS with both instruments is more efficient (smaller SE) than")
    print("  using either instrument alone -- it exploits more exogenous variation.")
    print("  All IV estimates are close to the true 0.10, while OLS is ~50% higher.")


# ==============================================================
# 5. First-Stage Diagnostics: Testing for Weak Instruments
# ==============================================================
# A weak instrument has a small effect on X.  The consequences:
#   - IV standard errors explode (IV is inconsistent in the limit).
#   - In finite samples, IV is biased TOWARD OLS when instruments are weak.
#   - Conventional t-tests become unreliable.
#
# Rule of thumb (Stock-Yogo 2005):
#   First-stage F > 10  -->  instrument is probably strong enough.
#   F < 10              -->  weak instrument; use LIML or weak-IV robust CIs.
#
# With multiple instruments: use the Cragg-Donald F or the Kleibergen-Paap
# rank statistic (robust to non-iid errors) to test joint relevance.

def first_stage_f(df: pd.DataFrame, x: str, instruments: list,
                  controls: list = None) -> float:
    """Return the F-statistic testing joint significance of instruments in the first stage."""
    controls = controls or []
    full_rhs  = " + ".join(instruments + controls)
    restr_rhs = " + ".join(controls) if controls else "1"
    full   = smf.ols(f"{x} ~ {full_rhs}",  data=df).fit()
    restr  = smf.ols(f"{x} ~ {restr_rhs}", data=df).fit()
    q  = len(instruments)
    n  = len(df)
    k  = full.df_model + 1
    f  = ((restr.ssr - full.ssr) / q) / (full.ssr / (n - k))
    return f


def demo_weak_instruments() -> None:
    """Show how first-stage F degrades as instrument strength weakens."""
    rng = np.random.default_rng(99)
    n   = N_OBS

    print(f"\n  Simulated data: vary the strength of a single binary instrument")
    print()
    print(f"  {'Effect of Z on educ':>20} | {'First-stage F':>14} | {'IV b_educ':>10} | Bias vs {TRUE_B_EDUC:.2f}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*10}-+-{'-'*12}")

    for delta_z in [2.0, 1.0, 0.5, 0.2, 0.05]:
        ability = rng.normal(0, 1, n)
        z       = rng.binomial(1, 0.5, n).astype(float)
        educ    = np.clip(12 + 0.8 * ability + delta_z * z + rng.normal(0, 1.5, n), 8, 20).round()
        eps     = rng.normal(0, 0.20, n)
        lw      = 1.5 + TRUE_B_EDUC * educ + TRUE_B_ABLT * ability + eps
        df2     = pd.DataFrame({"log_wage": lw, "educ": educ, "z": z})

        f_stat = first_stage_f(df2, "educ", ["z"])
        iv_res = fit_2sls(df2, "log_wage", "educ", instruments=["z"])

        print(f"  {delta_z:>20.2f} | {f_stat:>14.1f} | {iv_res['b']:>10.4f} | {iv_res['b'] - TRUE_B_EDUC:+.4f}")

    print()
    print("  As delta_z shrinks, the first-stage F drops below 10 and the IV")
    print("  estimate becomes unstable -- large variance AND bias toward OLS.")


# ==============================================================
# 6. The Hausman Endogeneity Test
# ==============================================================
# IV is more efficient than OLS when X is exogenous.  So the
# question is: do we actually need IV?
#
# Durbin-Wu-Hausman test:
#   H0: X is exogenous (OLS is consistent).
#   H_A: X is endogenous (OLS is inconsistent; IV is needed).
#
# Implementation via the auxiliary regression (Davidson-MacKinnon):
#   1. Regress X on Z to get residuals v̂.
#   2. Include v̂ as an extra regressor in the Y equation.
#   3. If b_v̂ is significant, X is endogenous.

def demo_hausman_test() -> None:
    df = make_iv_data()

    # Stage 1 residuals
    stage1 = smf.ols("educ ~ near_college + sibling_col", data=df).fit()
    df["v_hat"] = stage1.resid

    # Augmented regression
    aug = smf.ols("log_wage ~ educ + v_hat", data=df).fit()

    b_vhat = aug.params["v_hat"]
    se_vhat = aug.bse["v_hat"]
    p_vhat  = aug.pvalues["v_hat"]

    print(f"\n  Durbin-Wu-Hausman endogeneity test:")
    print(f"    H0: educ is exogenous (OLS consistent)")
    print(f"    Coefficient on first-stage residual (v_hat): {b_vhat:.4f}")
    print(f"    SE = {se_vhat:.4f}   t = {b_vhat / se_vhat:.2f}   p = {p_vhat:.4g}")
    print()
    if p_vhat < 0.05:
        print("    Reject H0 -- educ is endogenous.  IV is needed.")
    else:
        print("    Fail to reject H0 -- endogeneity not detected.  OLS may be fine.")
    print()
    print("  The residual v_hat captures the part of educ driven by ability.")
    print("  Its positive coefficient confirms that higher-ability workers get")
    print("  more schooling AND earn more -- the exact OVB mechanism at work.")


# ==============================================================
# 7. Overidentification: the Sargan-Hansen J-test
# ==============================================================
# With more instruments than endogenous variables, the model is
# overidentified.  The J-test (Sargan 1958, Hansen 1982) tests
# whether the extra instruments are valid:
#
#   H0: all instruments satisfy the exclusion restriction.
#   Statistic: J = n * R2 from regressing 2SLS residuals on all instruments.
#   Under H0: J ~ chi2(# overidentifying restrictions = # Z - # endog X).
#
# Rejecting H0 means at least one instrument is invalid -- but the
# test doesn't tell you which one.  Failure to reject doesn't prove
# validity; the test has limited power in small samples.

def demo_sargan_hansen() -> None:
    df = make_iv_data()

    # 2SLS with both instruments
    stage1 = smf.ols("educ ~ near_college + sibling_col", data=df).fit()
    df["educ_hat"] = stage1.fittedvalues

    stage2 = smf.ols("log_wage ~ educ_hat", data=df).fit()

    # 2SLS residuals: use actual X, not X_hat
    b_iv = stage2.params["educ_hat"]
    a_iv = stage2.params["Intercept"]
    df["e_iv"] = df["log_wage"] - a_iv - b_iv * df["educ"]

    # Sargan: J = n * R2 from regressing e_iv on all instruments
    sargan = smf.ols("e_iv ~ near_college + sibling_col", data=df).fit()
    j_stat = len(df) * sargan.rsquared
    p_j    = 1 - stats.chi2.cdf(j_stat, df=1)   # df = 2 instruments - 1 endog = 1

    print(f"\n  Sargan-Hansen J-test (overidentification test):")
    print(f"    H0: both instruments satisfy the exclusion restriction")
    print(f"    J = {j_stat:.3f}   chi2(1) p-value = {p_j:.4g}")
    print()
    if p_j < 0.05:
        print("    Reject H0 -- at least one instrument appears invalid.")
    else:
        print("    Fail to reject H0 -- instruments are consistent with exclusion.")
    print()
    print("  Passing the J-test provides some reassurance, but it cannot rule")
    print("  out the case where BOTH instruments are invalid in the same direction.")
    print("  Economic reasoning for the exclusion restriction is always primary.")


# ==============================================================
# 8. Practical Guide
# ==============================================================
# A checklist for building a credible IV design.

def demo_practical_guide() -> None:
    guide = [
        ("Is your treatment variable plausibly endogenous?",
         "No  -> use OLS; the Hausman test can help confirm this.",
         "Yes -> IV or another identification strategy is needed."),

        ("Do you have a candidate instrument?",
         "No  -> IV is not an option here; consider RD, DiD, or matching.",
         "Yes -> check relevance (first-stage F > 10) before proceeding."),

        ("Is the first-stage F > 10?",
         "Yes -> proceed to 2SLS.",
         "No  -> weak instrument.  Consider LIML or Anderson-Rubin CIs."),

        ("Can you defend the exclusion restriction?",
         "Yes -> state the economic argument clearly in your paper.",
         "No  -> the instrument is invalid; IV is worse than OLS."),

        ("Do you have more instruments than endogenous regressors?",
         "No (just-identified) -> J-test not applicable; exclusion untestable.",
         "Yes (overidentified) -> run the Sargan-Hansen J-test as a sanity check."),

        ("Are IV estimates sensitive to the choice of instrument?",
         "No  -> estimates are robust; report main spec.",
         "Yes -> investigate why; different IVs identify different LATEs."),
    ]

    print()
    for i, (question, opt_a, opt_b) in enumerate(guide, 1):
        print(f"  Step {i}: {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  The hardest part of IV is finding instruments that are both")
    print("  strong (relevant) and credibly excludable.  A weak but valid")
    print("  instrument is usable; a strong but invalid one is not.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. The Endogeneity Problem")
    demo_endogeneity_problem()

    section("2. Instrumental Variable Conditions")
    demo_iv_conditions()

    section("3. The Wald Estimator")
    demo_wald_estimator()

    section("4. Two-Stage Least Squares (2SLS)")
    demo_2sls()

    section("5. First-Stage Diagnostics: Weak Instruments")
    demo_weak_instruments()

    section("6. The Hausman Endogeneity Test")
    demo_hausman_test()

    section("7. Overidentification: the Sargan-Hansen J-test")
    demo_sargan_hansen()

    section("8. Practical Guide")
    demo_practical_guide()


if __name__ == "__main__":
    main()
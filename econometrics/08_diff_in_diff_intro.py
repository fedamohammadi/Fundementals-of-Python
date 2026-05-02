"""
Difference-in-Differences:
- Treatment / control setup and the 2x2 DiD estimator
- The parallel trends assumption and why it is not testable
- DiD with regression and adding controls
- Event study: dynamic treatment effects
- Staggered adoption: problems with the classic two-way FE approach
- Parallel trends placebo test (pre-treatment falsification)
- A practical guide for DiD designs
"""

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
# A simulated policy study: a new job-training programme rolled
# out in some states (treated) but not others (control) in 2019.
# We observe wages for N_STATES states over 6 years (2016–2021).
# Within each state a cross-section of N_PER_STATE workers is
# observed each year (repeated cross-section, not a true panel).
#
# Data generating process:
#   log_wage = state_fe + year_fe + b_treat * post * treated + eps
# where b_treat = TRUE_ATT is the Average Treatment Effect on
# the Treated (ATT) we want to recover.

N_STATES     = 40     # 20 treated, 20 control
N_PER_STATE  = 30     # workers per state-year
TRUE_ATT     = 0.08   # treatment raises wages by 8 log-points


def make_did_data(seed: int = 42,
                  parallel_trends: bool = True,
                  staggered: bool = False) -> pd.DataFrame:
    """
    Build a state x year panel.
    - parallel_trends=False: treated states have a pre-existing upward trend.
    - staggered=True: treatment adoption spreads across 2018-2020.
    """
    rng   = np.random.default_rng(seed)
    years = np.arange(2016, 2022)

    treated_states = np.arange(N_STATES // 2)
    control_states = np.arange(N_STATES // 2, N_STATES)

    state_fe = rng.normal(0, 0.15, N_STATES)

    if staggered:
        # Each treated state gets treatment in a random year between 2018-2020
        treat_year = {s: rng.integers(2018, 2021) for s in treated_states}
        treat_year.update({s: 9999 for s in control_states})  # never treated
    else:
        treat_year = {s: 2019 for s in treated_states}
        treat_year.update({s: 9999 for s in control_states})

    rows = []
    for s in range(N_STATES):
        treated = int(s in treated_states)
        for year in years:
            post   = int(year >= treat_year[s])
            n      = N_PER_STATE
            trend  = 0.0 if parallel_trends else (0.04 * (year - 2016) * treated)
            eps    = rng.normal(0, 0.10, n)
            log_wage = (state_fe[s]
                        + 0.03 * (year - 2016)  # common time trend
                        + trend
                        + TRUE_ATT * post
                        + eps)
            for w in range(n):
                rows.append({
                    "state":    s,
                    "year":     year,
                    "treated":  treated,
                    "post":     post,
                    "log_wage": log_wage[w],
                    "treat_year": treat_year[s],
                })

    return pd.DataFrame(rows)


# ==============================================================
# 1. The 2x2 DiD Estimator
# ==============================================================
# Classic DiD with two groups (treated/control) and two periods
# (pre/post).  The estimator is:
#
#   ATT_DiD = (ȳ_T_post - ȳ_T_pre) - (ȳ_C_post - ȳ_C_pre)
#
# The first difference removes pre-existing level differences
# between treated and control.  The second difference removes
# common time trends.  What remains is the treatment effect,
# provided the parallel trends assumption holds.

def demo_2x2_did() -> None:
    df   = make_did_data()
    pre  = df[df["year"] < 2019]
    post = df[df["year"] >= 2019]

    mean_t_pre  = pre [pre ["treated"] == 1]["log_wage"].mean()
    mean_c_pre  = pre [pre ["treated"] == 0]["log_wage"].mean()
    mean_t_post = post[post["treated"] == 1]["log_wage"].mean()
    mean_c_post = post[post["treated"] == 0]["log_wage"].mean()

    diff_treated = mean_t_post - mean_t_pre
    diff_control = mean_c_post - mean_c_pre
    did          = diff_treated - diff_control

    print(f"\n  True ATT = {TRUE_ATT:.3f}")
    print()
    print(f"  Group means:")
    print(f"    Treated  pre-treatment:  {mean_t_pre:.4f}")
    print(f"    Treated  post-treatment: {mean_t_post:.4f}  (change = {diff_treated:+.4f})")
    print(f"    Control  pre-treatment:  {mean_c_pre:.4f}")
    print(f"    Control  post-treatment: {mean_c_post:.4f}  (change = {diff_control:+.4f})")
    print()
    print(f"  DiD estimator = {diff_treated:.4f} - {diff_control:.4f} = {did:.4f}")
    print()
    print("  The control group's change is the counterfactual: how much would")
    print("  treated wages have risen WITHOUT the programme?  Subtracting it")
    print("  isolates the treatment-induced change.")


# ==============================================================
# 2. The Parallel Trends Assumption
# ==============================================================
# DiD identifies ATT under parallel trends: absent treatment,
# treated and control groups would have followed the same wage
# trajectory.  This is an ASSUMPTION, not a testable fact for
# the post-period.  It can be made more plausible by:
#   - Visual inspection of pre-treatment trends
#   - Comparing units that are similar on pre-treatment covariates
#   - Falsification tests using placebo outcomes or periods
#
# When parallel trends fails -- e.g., treated states had a
# pre-existing faster wage growth -- the DiD estimate is biased.

def demo_parallel_trends_failure() -> None:
    df_ok  = make_did_data(parallel_trends=True)
    df_bad = make_did_data(parallel_trends=False)

    print("\n  Comparing DiD estimates when parallel trends holds vs. fails:")
    print()
    print(f"  {'Dataset':>30} | {'DiD estimate':>13} | Bias vs. {TRUE_ATT:.3f}")
    print(f"  {'-'*30}-+-{'-'*13}-+-{'-'*15}")

    for label, df in [("Parallel trends holds", df_ok), ("Trend violation (+0.04/yr)", df_bad)]:
        pre  = df[df["year"] < 2019]
        post = df[df["year"] >= 2019]
        did  = ((post[post["treated"] == 1]["log_wage"].mean()
                 - pre [pre ["treated"] == 1]["log_wage"].mean())
                - (post[post["treated"] == 0]["log_wage"].mean()
                   - pre [pre ["treated"] == 0]["log_wage"].mean()))
        print(f"  {label:>30} | {did:>13.4f} | {did - TRUE_ATT:+.4f}")

    print()
    print("  A pre-existing upward trend in the treated group inflates the DiD")
    print("  estimate: the extra post-treatment wage growth is partially a")
    print("  continuation of the pre-trend, not the treatment effect.")


# ==============================================================
# 3. DiD with Regression and Controls
# ==============================================================
# The regression form of DiD is:
#   y_st = a + b1*treated_s + b2*post_t + b3*(treated_s * post_t) + eps_st
#
# b3 is the DiD estimator (= ATT under parallel trends).
# This generalises the 2x2 table to:
#   - Multiple pre and post periods
#   - Adding state and year fixed effects (absorbs b1 and b2)
#   - Including worker-level controls
#   - Proper standard errors clustered at the state level
#
# With state and year FEs the estimating equation becomes:
#   y_ist = alpha_s + gamma_t + b*(treated_s * post_st) + X_ist'c + eps_ist

def demo_did_regression() -> None:
    df = make_did_data()
    # post_common is 1 for everyone in years >= 2019 (common time indicator)
    df["post_common"] = (df["year"] >= 2019).astype(int)
    df["treat_post"]  = df["treated"] * df["post_common"]

    # Bare-bones 2x2 form
    m1 = smf.ols("log_wage ~ treated + post_common + treat_post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]}
    )

    # With state and year fixed effects (preferred)
    m2 = smf.ols("log_wage ~ treat_post + C(state) + C(year)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]}
    )

    print(f"\n  True ATT = {TRUE_ATT:.3f}")
    print()
    print(f"  {'Specification':>32} | {'b_treat_post':>13} | {'SE':>7} | {'p':>7}")
    print(f"  {'-'*32}-+-{'-'*13}-+-{'-'*7}-+-{'-'*7}")

    for label, res in [
        ("No FEs (treated + post + treat_post)", m1),
        ("State + year FEs",                     m2),
    ]:
        b  = res.params["treat_post"]
        se = res.bse["treat_post"]
        p  = res.pvalues["treat_post"]
        print(f"  {label:>32} | {b:>13.4f} | {se:>7.4f} | {p:>7.4f}")

    print()
    print("  Adding state and year FEs absorbs persistent state-level differences")
    print("  and common year shocks, making the treatment effect estimator more")
    print("  credible.  Cluster SEs at the state level -- the treatment assignment")
    print("  varies at the state level, not the worker level.")


# ==============================================================
# 4. Event Study: Dynamic Treatment Effects
# ==============================================================
# A static DiD collapses all post-treatment periods into a single
# "post" indicator.  An event study instead estimates a separate
# treatment effect coefficient for each relative year:
#
#   y_ist = alpha_s + gamma_t + sum_k b_k * 1(t - t_s* = k) + eps_ist
#
# where t_s* is state s's treatment year and k indexes time since
# treatment (negative k = pre-treatment, k=0 = treatment year).
#
# Three things to check in an event study plot:
#   1. Pre-period coefficients near zero (parallel pre-trends)
#   2. Treatment effect appears at k=0, not earlier (no anticipation)
#   3. Effect size and persistence post-treatment

def _rk_name(k: int) -> str:
    """Column name for relative-year k (avoids '-' in patsy formula)."""
    return f"rkm{abs(k)}" if k < 0 else f"rkp{k}"


def demo_event_study() -> None:
    df = make_did_data()
    df["rel_year"] = df["year"] - df["treat_year"]
    df.loc[df["treated"] == 0, "rel_year"] = -99   # control group never treated

    # Dummies for k = -3, -2 (pre), 0, 1, 2 (post); omit k = -1 as reference
    rel_years = [-3, -2, 0, 1, 2]
    for k in rel_years:
        df[_rk_name(k)] = ((df["rel_year"] == k) & (df["treated"] == 1)).astype(float)

    rk_terms = " + ".join(_rk_name(k) for k in rel_years)
    formula  = f"log_wage ~ {rk_terms} + C(state) + C(year)"

    res = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]}
    )

    print(f"\n  True ATT = {TRUE_ATT:.3f}  (same each post-period by construction)")
    print()
    print(f"  {'Relative year k':>18} | {'Coefficient':>12} | {'SE':>7} | {'p':>7} | Period")
    print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

    for k in rel_years:
        col = _rk_name(k)
        b   = res.params.get(col, 0)
        se  = res.bse.get(col, 0)
        p   = res.pvalues.get(col, 1)
        period = "pre" if k < 0 else "post"
        print(f"  {k:>18} | {b:>12.4f} | {se:>7.4f} | {p:>7.4f} | {period}")

    print()
    print("  (k = -1 is the omitted reference year.  Pre-period coefficients")
    print("  should be close to zero if parallel trends holds.  Post-period")
    print("  coefficients estimate the treatment effect in each year.)")


# ==============================================================
# 5. Staggered Adoption and TWFE Problems
# ==============================================================
# When units adopt treatment at different times (staggered rollout),
# the standard two-way FE (TWFE) DiD has a subtle flaw.
#
# TWFE uses already-treated units as implicit controls for later-
# treated units.  If treatment effects are heterogeneous (different
# units have different ATTs), the TWFE estimator can be a
# weighted average with NEGATIVE weights on some groups -- it can
# even have the wrong sign.
#
# Here we show the problem by simulating staggered adoption and
# comparing TWFE to the manual 2x2 estimate for early vs. late
# adopters separately.

def demo_staggered_did() -> None:
    df = make_did_data(staggered=True)
    df["treat_post"] = df["treated"] * df["post"]

    # TWFE estimate
    twfe = smf.ols("log_wage ~ treat_post + C(state) + C(year)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]}
    )

    # Manual 2x2 for early adopters (treated in 2018) vs. control only
    early = df[(df["treat_year"] == 2018) | (df["treated"] == 0)].copy()
    early["treat_post_e"] = early["treated"] * (early["year"] >= 2018).astype(int)
    twfe_e = smf.ols("log_wage ~ treat_post_e + C(state) + C(year)", data=early).fit(
        cov_type="cluster", cov_kwds={"groups": early["state"]}
    )

    print(f"\n  True ATT = {TRUE_ATT:.3f}  (same for all states by construction)")
    print()
    print(f"  {'Estimator':>35} | {'Estimate':>9} | Bias")
    print(f"  {'-'*35}-+-{'-'*9}-+-{'-'*15}")
    print(f"  {'TWFE (staggered, all treated)':>35} | {twfe.params['treat_post']:>9.4f} | "
          f"{twfe.params['treat_post'] - TRUE_ATT:+.4f}")
    print(f"  {'2x2 early adopters vs. control':>35} | {twfe_e.params['treat_post_e']:>9.4f} | "
          f"{twfe_e.params['treat_post_e'] - TRUE_ATT:+.4f}")
    print()
    print("  TWFE can differ from the true ATT in staggered designs because")
    print("  it implicitly compares later-treated units against already-treated")
    print("  units.  When effects are homogeneous (as here), bias is mild;")
    print("  it can be severe with heterogeneous effects.  Callaway-Sant'Anna")
    print("  or stacked regressions are preferred in staggered settings.")


# ==============================================================
# 6. Parallel Trends Placebo Test
# ==============================================================
# The key assumption in DiD -- that treated and control groups
# would have trended the same without treatment -- is not directly
# testable for the post period.  But we CAN test it for pre-periods
# by pretending an earlier year was the "treatment year" and
# running the DiD on pre-treatment data only.
#
# If the placebo DiD is significantly different from zero, parallel
# trends was already broken before the real treatment arrived --
# a red flag for the whole design.

def demo_placebo_test() -> None:
    df     = make_did_data(parallel_trends=True)
    df_bad = make_did_data(parallel_trends=False)

    print(f"\n  Placebo test: pretend treatment happened in 2017 (true year: 2019)")
    print(f"  Using pre-treatment data only (2016 and 2017).")
    print()
    print(f"  {'Dataset':>30} | {'Placebo DiD':>12} | {'SE':>7} | {'p':>7} | Verdict")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")

    for label, df_ in [("Parallel trends holds", df), ("Trend violation (+0.04/yr)", df_bad)]:
        pre_only = df_[df_["year"] <= 2017].copy()
        pre_only["placebo_post"]    = (pre_only["year"] == 2017).astype(int)
        pre_only["treat_placebo"]   = pre_only["treated"] * pre_only["placebo_post"]

        res = smf.ols(
            "log_wage ~ treat_placebo + C(state) + C(year)",
            data=pre_only
        ).fit(cov_type="cluster", cov_kwds={"groups": pre_only["state"]})

        b = res.params["treat_placebo"]
        se = res.bse["treat_placebo"]
        p  = res.pvalues["treat_placebo"]
        verdict = "FAIL -- pre-trend detected" if p < 0.10 else "pass"
        print(f"  {label:>30} | {b:>12.4f} | {se:>7.4f} | {p:>7.4f} | {verdict}")

    print()
    print("  When trends genuinely differ pre-treatment, the placebo DiD is")
    print("  large and significant -- giving a way to falsify your design")
    print("  before you look at post-treatment outcomes.")


# ==============================================================
# 7. Practical Guide
# ==============================================================
# A checklist for building a credible DiD design.

def demo_practical_guide() -> None:
    guide = [
        ("Is there a clear pre-treatment period for both groups?",
         "No  -> DiD is not applicable; consider synthetic control or RD.",
         "Yes -> continue."),

        ("Does treatment happen at the same time for all treated units?",
         "Yes -> standard 2x2 or TWFE DiD.",
         "No (staggered) -> use Callaway-Sant'Anna or stacked regressions instead of TWFE."),

        ("Do pre-treatment trends look parallel (visual + placebo test)?",
         "Yes -> DiD assumption is supported (not proven).",
         "No  -> DiD is not valid.  Consider matching or synthetic control."),

        ("Are the treated and control groups similar on observables?",
         "Yes -> basic DiD is fine.",
         "No  -> add controls or use a doubly-robust DiD estimator."),

        ("Cluster level for standard errors?",
         "Cluster at the level where treatment varies (usually state/city/firm).",
         "Larger clusters = fewer clusters = use wild cluster bootstrap if <30 groups."),
    ]

    print()
    for i, (question, opt_a, opt_b) in enumerate(guide, 1):
        print(f"  Step {i}: {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  The hardest part of DiD is finding a good control group --")
    print("  units that would have followed the treated group's counterfactual")
    print("  path absent the intervention.  More controls is not always better;")
    print("  similar controls are what matter.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. The 2x2 DiD Estimator")
    demo_2x2_did()

    section("2. The Parallel Trends Assumption")
    demo_parallel_trends_failure()

    section("3. DiD with Regression and Controls")
    demo_did_regression()

    section("4. Event Study: Dynamic Treatment Effects")
    demo_event_study()

    section("5. Staggered Adoption and TWFE Problems")
    demo_staggered_did()

    section("6. Parallel Trends Placebo Test")
    demo_placebo_test()

    section("7. Practical Guide")
    demo_practical_guide()


if __name__ == "__main__":
    main()

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
            trend  = 0.0 if parallel_trends else (0.015 * (year - 2016) * treated)
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

    for label, df in [("Parallel trends holds", df_ok), ("Trend violation (+0.015/yr)", df_bad)]:
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

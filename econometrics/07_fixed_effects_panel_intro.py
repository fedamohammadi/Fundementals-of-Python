"""
Fixed Effects and Panel Data:
- Panel data structure: balanced vs. unbalanced panels
- Pooled OLS and omitted variable bias from unobserved heterogeneity
- The within estimator: entity demeaning and fixed effects
- Two-way fixed effects: entity and time fixed effects
- Fixed effects vs. first differences
- Testing for fixed effects: joint F-test and Mundlak-Hausman
- What FE cannot fix: time-invariant variables and measurement error
- A practical guide for choosing between panel estimators
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared data generation
# ==============================================================
# Workers observed for N_YEARS years.
# Key time-varying outcome: whether a worker is covered by a
# union contract ("union").  Union membership is correlated with
# unobserved ability -- less productive workers seek union
# protection -- so pooled OLS underestimates the true union
# wage premium.  FE identifies it from workers who switch
# in or out of union coverage across years.

N_WORKERS   = 200
N_YEARS     = 6
TRUE_B_UNION = 0.15   # union wage premium (15 log-points)
TRUE_B_EXPER = 0.006  # wage return per year of experience


def make_panel(seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    years = np.arange(2015, 2015 + N_YEARS)

    # Worker-level (time-invariant)
    ability = rng.normal(0, 0.35, N_WORKERS)
    educ    = np.clip(12 + 2 * ability + rng.normal(0, 1.5, N_WORKERS), 8, 20).round()
    female  = rng.binomial(1, 0.45, N_WORKERS).astype(float)
    base_xp = rng.integers(0, 10, N_WORKERS)  # starting experience differs across workers

    # Small aggregate year shocks (absorbed by year dummies in two-way FE)
    year_shock = {y: rng.normal(0, 0.04) for y in years}

    rows = []
    for t, year in enumerate(years):
        exper = (base_xp + t).astype(float)

        # Union membership: less able workers more likely in unions (negative selection)
        union_logit = -0.6 * ability + rng.normal(0, 0.9, N_WORKERS)
        union_prob  = 1 / (1 + np.exp(-union_logit))
        union       = rng.binomial(1, union_prob, N_WORKERS).astype(float)

        eps = rng.normal(0, 0.12, N_WORKERS)
        log_wage = (1.6
                    + TRUE_B_UNION * union
                    + TRUE_B_EXPER * exper
                    + 0.35 * ability
                    + year_shock[year]
                    + eps)

        for i in range(N_WORKERS):
            rows.append({
                "worker_id": i,
                "year":      year,
                "log_wage":  log_wage[i],
                "union":     union[i],
                "exper":     exper[i],
                "educ":      educ[i],
                "female":    female[i],
                "ability":   ability[i],
            })

    return pd.DataFrame(rows).sort_values(["worker_id", "year"]).reset_index(drop=True)


# ==============================================================
# 1. Panel Data Structure
# ==============================================================
# A panel has two indices: entity i and time t.
# Balanced means every entity appears in every period.
#
# Key diagnostic before estimation: decompose variance into
# "between" (variation across units) and "within" (variation
# for the same unit over time).  FE uses only within variation.
# If a variable barely changes within units, FE estimates its
# effect imprecisely even if its between-variation is large.

def demo_panel_structure() -> None:
    df  = make_panel()
    n_i = df["worker_id"].nunique()
    n_t = df["year"].nunique()

    switchers = (df.groupby("worker_id")["union"].nunique() > 1).sum()

    print(f"\n  Panel: {n_i} workers x {n_t} years = {len(df)} observations")
    print(f"  Years: {df['year'].min()} – {df['year'].max()}")
    print(f"  Balanced: {len(df) == n_i * n_t}")
    print(f"  Workers who switch union status at least once: {switchers} / {n_i}")
    print()
    print(f"  {'Variable':>9} | {'Overall SD':>11} | {'Between SD':>11} | {'Within SD':>10}")
    print(f"  {'-'*9}-+-{'-'*11}-+-{'-'*11}-+-{'-'*10}")

    for var in ["log_wage", "union", "exper"]:
        overall = df[var].std()
        between = df.groupby("worker_id")[var].mean().std()
        within  = (df[var]
                   - df.groupby("worker_id")[var].transform("mean")
                   + df[var].mean()).std()
        print(f"  {var:>9} | {overall:>11.4f} | {between:>11.4f} | {within:>10.4f}")

    print()
    print("  FE exploits within-worker variation.  'union' changes for many")
    print("  workers across years, so there is real within-variation to use.")
    print("  Workers who never switch don't contribute to the FE estimate.")


# ==============================================================
# 2. Pooled OLS and Omitted Variable Bias
# ==============================================================
# Omitting ability biases OLS when ability is correlated with the
# treatment.  Here ability is negatively correlated with union
# membership (less productive workers cluster in unions), so OLS
# picks up a spurious negative ability effect and underestimates
# the true union wage premium.
#
# OVB formula:
#   plim(b_OLS) = b_true + (cov(union, ability) / var(union)) * b_ability
# cov(union, ability) < 0, b_ability > 0 --> bias is negative,
# so OLS understates the true union premium.

def demo_pooled_ols_bias() -> None:
    df  = make_panel()
    res = smf.ols("log_wage ~ union + exper", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["worker_id"]}
    )

    print(f"\n  True: b_union = {TRUE_B_UNION:.3f}   b_exper = {TRUE_B_EXPER:.4f}")
    print()
    print(f"  Pooled OLS (clustered SEs):")
    for param in ["union", "exper"]:
        b    = res.params[param]
        se   = res.bse[param]
        true = TRUE_B_UNION if param == "union" else TRUE_B_EXPER
        print(f"    b_{param:<5} = {b:.4f}   SE = {se:.4f}   bias = {b - true:+.4f}")

    corr = df[["union", "ability"]].corr().loc["union", "ability"]
    print()
    print(f"  Corr(union, ability) = {corr:.3f}  (negative selection into unions)")
    print("  Less productive workers are more likely in unions, pulling the OLS")
    print("  estimate of the union premium toward zero.")


# ==============================================================
# 3. The Within Estimator (Entity Fixed Effects)
# ==============================================================
# FE demeans each entity's data before regressing:
#   ỹ_it = y_it - ȳ_i      ũ_it = union_it - ū_i
#
# The entity mean absorbs any time-invariant term, including ability.
# OLS on demeaned data is the within estimator.
#
# LSDV (Least Squares Dummy Variables): equivalent to adding a
# dummy for each entity.  Numerically identical slopes to demeaning,
# but slow for large N.

def demean(df: pd.DataFrame, var: str, group: str) -> pd.Series:
    return df[var] - df.groupby(group)[var].transform("mean")


def demo_within_estimator() -> None:
    df = make_panel()

    df["log_wage_dm"] = demean(df, "log_wage", "worker_id")
    df["union_dm"]    = demean(df, "union",    "worker_id")
    df["exper_dm"]    = demean(df, "exper",    "worker_id")

    fe_manual = smf.ols("log_wage_dm ~ union_dm + exper_dm - 1", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["worker_id"]}
    )
    lsdv   = smf.ols("log_wage ~ union + exper + C(worker_id)", data=df).fit()
    pooled = smf.ols("log_wage ~ union + exper",                data=df).fit()

    print(f"\n  True: b_union = {TRUE_B_UNION:.3f}   b_exper = {TRUE_B_EXPER:.4f}")
    print()
    print(f"  {'Estimator':>15} | {'b_union':>8} | {'b_exper':>9} | Note")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*9}-+-{'-'*32}")

    for label, res, uk, xk in [
        ("Pooled OLS",  pooled,    "union",    "exper"),
        ("FE (demean)", fe_manual, "union_dm", "exper_dm"),
        ("LSDV",        lsdv,      "union",    "exper"),
    ]:
        print(f"  {label:>15} | {res.params[uk]:>8.4f} | {res.params[xk]:>9.4f} | "
              f"{'biased (ability omitted)' if label == 'Pooled OLS' else 'ability absorbed'}")

    print()
    print("  FE recovers the true union premium by using only within-worker")
    print("  changes: when a worker joins or leaves a union, does their wage")
    print("  change by the predicted amount?  Ability is constant across that")
    print("  comparison and drops out automatically.")

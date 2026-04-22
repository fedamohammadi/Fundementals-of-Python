"""
Robust and Clustered Standard Errors:
- Why OLS standard errors are wrong under heteroskedasticity
- Heteroskedasticity-consistent (HC) robust standard errors
- Intra-cluster correlation and why it inflates false-positive rates
- Clustered standard errors: implementation and interpretation
- Comparing OLS, HC-robust, and clustered SEs side by side
- A practical decision guide for choosing the right SE type
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
# We build a dataset of workers nested in firms.
# This lets us demonstrate both heteroskedasticity (variance grows
# with firm size) and clustering (workers in the same firm share
# unobserved shocks, making their residuals correlated).

N_FIRMS   = 50
N_WORKERS = 20    # workers per firm  ->  n = 1000 total


def make_data(seed: int = 42) -> pd.DataFrame:
    """
    Simulate a wage dataset with:
      - firm-level unobserved heterogeneity (shared shock per firm)
      - worker-level heteroskedasticity (variance grows with educ)
    """
    rng = np.random.default_rng(seed)

    firm_id     = np.repeat(np.arange(N_FIRMS), N_WORKERS)
    firm_shock  = np.repeat(rng.normal(0, 0.3, N_FIRMS), N_WORKERS)
    firm_size   = np.repeat(rng.integers(20, 500, N_FIRMS), N_WORKERS).astype(float)

    educ  = rng.integers(8, 22, size=N_FIRMS * N_WORKERS).astype(float)
    exper = rng.integers(0, 41, size=N_FIRMS * N_WORKERS).astype(float)

    # Heteroskedastic noise: variance scales with education
    error_std = 0.05 * educ
    eps       = firm_shock + rng.normal(0, error_std, size=N_FIRMS * N_WORKERS)

    log_wage = 1.6 + 0.10 * educ + 0.008 * exper + eps

    return pd.DataFrame({
        "log_wage":  log_wage,
        "educ":      educ,
        "exper":     exper,
        "firm_id":   firm_id,
        "firm_size": firm_size,
    })


# ==============================================================
# 1. Why OLS Standard Errors Can Be Wrong
# ==============================================================
# The OLS formula for standard errors assumes homoskedastic errors:
#   Var[b_OLS] = sigma^2 * (X'X)^-1
#
# When errors are heteroskedastic (Var[eps_i] != sigma^2), this
# formula is wrong. Reported SEs are biased, so t-statistics and
# p-values are invalid even though the OLS point estimates are fine.
#
# This section shows the bias by comparing the OLS SE against the
# true sampling variability of b across many simulated datasets.

def demo_ols_se_bias() -> None:
    """
    Re-simulate 500 datasets and record the OLS estimate of b_educ
    each time. Compare the true sampling std of b_educ to the SE
    that OLS reports -- the gap reveals the bias.
    """
    n_sims   = 500
    b_draws  = np.zeros(n_sims)
    se_draws = np.zeros(n_sims)

    for i in range(n_sims):
        df  = make_data(seed=i)
        res = smf.ols("log_wage ~ educ + exper", data=df).fit()
        b_draws[i]  = res.params["educ"]
        se_draws[i] = res.bse["educ"]

    true_sampling_std = b_draws.std()
    avg_ols_se        = se_draws.mean()

    print(f"\n  Results across {n_sims} simulated datasets:")
    print(f"    True sampling std of b_educ : {true_sampling_std:.5f}")
    print(f"    Average OLS-reported SE     : {avg_ols_se:.5f}")
    print(f"    Ratio (OLS SE / true std)   : {avg_ols_se / true_sampling_std:.3f}")
    print()
    print("  A ratio below 1.0 means OLS underestimates uncertainty.")
    print("  This leads to t-statistics that are too large and p-values")
    print("  that are too small -- generating false positives.")


# ==============================================================
# 2. Heteroskedasticity-Consistent (HC) Robust Standard Errors
# ==============================================================
# The HC sandwich estimator replaces sigma^2 with the squared
# residuals e_i^2 in the variance formula:
#   Var_HC[b] = (X'X)^-1 * (sum of x_i x_i' e_i^2) * (X'X)^-1
#
# Common flavours (the differences are finite-sample corrections):
#   HC0: no correction       -- biased downward in small samples
#   HC1: multiplied by n/(n-k-1)  -- most commonly used
#   HC3: uses (e_i/(1-h_ii))^2   -- conservative; preferred when n is small
#
# In statsmodels: result.get_robustcov_results(cov_type='HC1')
# or pass cov_type='HC1' directly to .fit().

def compare_se_flavours(result_ols) -> None:
    """Print OLS, HC0, HC1, and HC3 SEs side by side for educ and exper."""
    rows = {}
    for cov in ["nonrobust", "HC0", "HC1", "HC3"]:
        label = "OLS" if cov == "nonrobust" else cov
        res   = result_ols.get_robustcov_results(cov_type=cov)
        rows[label] = {
            "educ_se":  res.bse["educ"],
            "exper_se": res.bse["exper"],
            "educ_t":   res.tvalues["educ"],
            "educ_p":   res.pvalues["educ"],
        }

    print(f"\n  {'SE type':>8} | {'SE(educ)':>10} | {'SE(exper)':>10} | {'t(educ)':>9} | {'p(educ)':>9}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")
    for label, r in rows.items():
        print(f"  {label:>8} | {r['educ_se']:>10.5f} | {r['exper_se']:>10.5f} | "
              f"{r['educ_t']:>9.3f} | {r['educ_p']:>9.4f}")


def demo_hc_robust_se() -> None:
    """
    Fit OLS on the heteroskedastic dataset and compare how much
    the SE changes when we switch to HC-robust corrections.
    Larger SEs under HC = OLS was understating uncertainty.
    """
    df     = make_data()
    result = smf.ols("log_wage ~ educ + exper", data=df).fit()

    print(f"\n  OLS point estimates (unchanged by SE correction):")
    print(f"    b_educ  = {result.params['educ']:.5f}")
    print(f"    b_exper = {result.params['exper']:.5f}")
    print(f"  R2 = {result.rsquared:.4f}")

    compare_se_flavours(result)

    print()
    print("  HC1 and HC3 SEs are larger than OLS -- correct direction.")
    print("  The point estimates do not change; only the uncertainty does.")
    print("  Use HC3 in small samples; HC1 is standard in most applied work.")


# ==============================================================
# 3. Intra-Cluster Correlation (ICC)
# ==============================================================
# When observations are grouped (students in schools, workers in
# firms, patients in hospitals), residuals within the same group
# are often correlated. OLS treats all observations as independent,
# so it overcounts the effective sample size and produces SEs that
# are too small.
#
# The intra-cluster correlation (ICC) measures how similar
# observations within a cluster are relative to the overall variance:
#   ICC = variance_between_clusters / total_variance
#
# ICC = 0  ->  no clustering problem; OLS SEs are fine.
# ICC > 0  ->  observations within a cluster carry less new information
#              than OLS assumes; standard SEs are too optimistic.

def compute_icc(residuals: pd.Series, cluster_id: pd.Series) -> float:
    """
    Estimate ICC using a one-way ANOVA decomposition of the residuals.
    Returns the fraction of total variance explained by cluster membership.
    """
    overall_mean = residuals.mean()
    cluster_means = residuals.groupby(cluster_id).mean()
    cluster_sizes = residuals.groupby(cluster_id).count()

    ss_between = (cluster_sizes * (cluster_means - overall_mean) ** 2).sum()
    ss_total   = ((residuals - overall_mean) ** 2).sum()

    return float(ss_between / ss_total)


def demo_icc() -> None:
    """
    Show that the firm-level shock in our dataset creates meaningful
    within-firm correlation, which is exactly what clustered SEs correct for.
    Also show that ICC rises as the firm shock variance increases.
    """
    df     = make_data()
    result = smf.ols("log_wage ~ educ + exper", data=df).fit()

    icc = compute_icc(result.resid, df["firm_id"])

    print(f"\n  Dataset: {N_FIRMS} firms x {N_WORKERS} workers = {N_FIRMS * N_WORKERS} obs")
    print(f"  Estimated ICC of OLS residuals: {icc:.4f}")
    print()
    print("  Interpretation:")
    if icc < 0.05:
        print("  ICC < 0.05: mild clustering. HC robust SEs may be sufficient.")
    elif icc < 0.20:
        print("  ICC in 0.05-0.20: moderate clustering. Clustered SEs recommended.")
    else:
        print("  ICC > 0.20: strong clustering. Clustered SEs are essential.")

    print()
    print("  Rule of thumb: if you can think of a reason why two observations")
    print("  share a common unobserved factor, use clustered SEs.")
    print("  Examples: students in the same class, states under the same policy,")
    print("  repeated observations on the same individual over time.")


# ==============================================================
# 4. Clustered Standard Errors
# ==============================================================
# Clustered SEs are a generalisation of HC robust SEs where the
# 'meat' of the sandwich sums over clusters rather than individuals:
#   Var_cluster[b] = (X'X)^-1 * (sum_g X_g' e_g e_g' X_g) * (X'X)^-1
#
# where g indexes clusters and e_g is the vector of residuals for
# cluster g. This allows arbitrary correlation within each cluster
# while assuming independence across clusters.
#
# In statsmodels: fit(cov_type='cluster', cov_kwds={'groups': cluster_var})
#
# Key requirement: enough clusters. With fewer than ~30-50 clusters
# the cluster-robust variance estimator itself becomes unreliable.
# The wild cluster bootstrap is preferred in small-cluster settings.

def demo_clustered_se() -> None:
    """
    Fit OLS with and without clustered SEs (clustering on firm_id).
    Compare the SE, t-statistic, and p-value for b_educ to show how
    much clustering matters once within-firm correlation is present.
    """
    df = make_data()

    fit_ols      = smf.ols("log_wage ~ educ + exper", data=df).fit()
    fit_hc1      = fit_ols.get_robustcov_results(cov_type="HC1")
    fit_cluster  = fit_ols.get_robustcov_results(
        cov_type="cluster",
        groups=df["firm_id"],
    )

    print(f"\n  b_educ = {fit_ols.params['educ']:.5f}  (same across all SE types)")
    print()
    print(f"  {'SE type':>15} | {'SE(educ)':>10} | {'t(educ)':>9} | {'p(educ)':>9} | 95% CI")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*22}")

    for label, res in [("OLS", fit_ols), ("HC1 robust", fit_hc1), ("Clustered", fit_cluster)]:
        se  = res.bse["educ"]
        t   = res.tvalues["educ"]
        p   = res.pvalues["educ"]
        ci  = res.conf_int().loc["educ"]
        print(f"  {label:>15} | {se:>10.5f} | {t:>9.3f} | {p:>9.4f} | "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    print()
    print("  The clustered SE is the largest of the three because it accounts")
    print("  for both heteroskedasticity AND within-firm correlation.")
    print("  If the clustered SE is close to OLS, clustering may not matter much.")

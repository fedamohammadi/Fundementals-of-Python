"""
Logit and Probit: Binary Choice Models:
- Why OLS fails for binary outcomes (Linear Probability Model problems)
- The latent variable model underlying discrete choice
- Logit: logistic CDF, log-odds interpretation
- Probit: normal CDF and coefficient comparison
- Marginal effects: average (AME) vs at the mean (MEM)
- Model fit: pseudo-R², AIC/BIC, and classification accuracy
- Predicted probabilities and decision threshold selection
- A practical guide for binary outcome models
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
# Labor force participation decision (Mroz 1987 style).
# A woman participates if her latent utility exceeds zero.
# True latent index:
#   y* = -2.0 + 0.08*educ + 0.05*exper - 0.001*exper² - 0.80*kids + eps
#   y  = 1 if y* > 0, else 0
#
# The error is logistic by construction, making logit the true model.

N_OBS = 1500


def make_binary_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    educ  = rng.integers(8, 19, N_OBS).astype(float)
    exper = rng.uniform(0, 30, N_OBS)
    kids  = rng.binomial(1, 0.40, N_OBS).astype(float)

    latent = (-2.0
              + 0.08  * educ
              + 0.05  * exper
              - 0.001 * exper ** 2
              - 0.80  * kids
              + rng.logistic(0, 1, N_OBS))

    return pd.DataFrame({
        "participate": (latent > 0).astype(int),
        "educ":        educ,
        "exper":       exper,
        "exper2":      exper ** 2,
        "kids":        kids,
    })


# ==============================================================
# 1. Why OLS Fails for Binary Outcomes
# ==============================================================
# The Linear Probability Model (LPM) is OLS applied to a 0/1 outcome.
# It has three structural problems:
#
#   1. Predicted probabilities outside [0, 1]: fitted values can be
#      negative or greater than 1, which is nonsensical as a probability.
#
#   2. Heteroskedastic errors by construction: Var(eps | x) = p(x)[1-p(x)]
#      varies with x.  OLS SEs are inconsistent unless robust SEs are used.
#
#   3. Non-linear truth: P(y=1 | x) is an S-shaped curve bounded in [0,1].
#      OLS forces a linear approximation that is only accurate in the middle.

def demo_lpm_problems() -> None:
    df  = make_binary_data()
    lpm = smf.ols("participate ~ educ + exper + exper2 + kids", data=df).fit()

    phat    = lpm.fittedvalues
    n_out   = ((phat < 0) | (phat > 1)).sum()
    pct_out = n_out / len(df) * 100

    print(f"\n  Participation rate: {df['participate'].mean():.3f}  |  LPM R² = {lpm.rsquared:.4f}")
    print()
    print(f"  Predicted probabilities outside [0, 1]: {n_out} ({pct_out:.1f}%)")
    print(f"    Min fitted value: {phat.min():.4f}")
    print(f"    Max fitted value: {phat.max():.4f}")
    print()
    print(f"  LPM coefficient on educ: {lpm.params['educ']:.4f}  SE = {lpm.bse['educ']:.4f}")
    print()
    print("  Interpretation: one more year of education raises P(participate)")
    print("  by a constant amount regardless of current education level.")
    print("  This is implausible at the tails of the distribution.")
    print()
    print("  LPM is a fast first pass but logit/probit respect the [0,1] constraint.")


# ==============================================================
# 2. The Latent Variable Model
# ==============================================================
# Both logit and probit derive from the same latent-index setup:
#
#   y*  = xβ + eps   (unobserved utility or propensity)
#   y   = 1{y* > 0}
#
# If eps ~ Logistic(0, π²/3): logit model.
# If eps ~ Normal(0, 1):      probit model.
#
# Scale is not identified (we only see y = 1{y* > 0}, not |y*|), so
# coefficients in probit are normalized to Var(eps) = 1.  In logit,
# Var(eps) = π²/3 ≈ 3.29.  This is why:
#
#   β_logit ≈ β_probit × (π / √3) ≈ 1.81 × β_probit
#
# P(y=1 | x) = F(xβ), where F is the model CDF.

def demo_latent_variable() -> None:
    scale = np.pi / np.sqrt(3)   # ≈ 1.8138

    print(f"\n  CDF comparison: P(y=1 | xβ) for logit vs probit")
    print()
    print(f"  {'xβ':>6} | {'P logit':>10} | {'P probit':>10} | Difference")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for xb in [-3, -2, -1, 0, 1, 2, 3]:
        pl = 1 / (1 + np.exp(-xb))
        pp = stats.norm.cdf(xb)
        print(f"  {xb:>6.1f} | {pl:>10.4f} | {pp:>10.4f} | {pl - pp:>+10.4f}")

    print()
    print(f"  Scaling factor π/√3 ≈ {scale:.4f}:")
    print("    β_logit ≈ 1.81 × β_probit")
    print("    Predicted probabilities from both models are nearly identical.")
    print()
    print("  The two CDFs agree closely in the center and diverge slightly")
    print("  in the tails (logit has heavier tails than the normal).")
    print("  In practice the choice between logit and probit rarely changes conclusions.")

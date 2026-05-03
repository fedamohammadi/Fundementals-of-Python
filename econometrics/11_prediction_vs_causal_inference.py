"""
Prediction vs. Causal Inference:
- The core distinction: predicting y vs. estimating the effect of x on y
- The bias-variance tradeoff: overfitting and underfitting
- Cross-validation: selecting model complexity by out-of-sample performance
- Regularization: Ridge and Lasso for high-dimensional prediction
- Why good prediction does not imply causal identification
- Bad controls: post-treatment variables and the Table 2 fallacy
- A practical guide for choosing prediction vs. causal approaches
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
# Dataset 1 (prediction): wages with many features, most are noise.
#   True model: y = 2*x1 + 1.5*x2 + 0.5*x3 + eps  (x4...x20 are noise)
#
# Dataset 2 (causal): treatment effect with confounding and mediation.
#   True total effect of treat on y is 2.0.
#   confound -> treat  (introduces OVB when omitted)
#   treat    -> mediator  (mediator is a bad control when included)
#
#   y = 2.0*treat + 3.0*confound + eps_y
#   mediator = 0.6*treat + eps_m

N_PRED   = 600
N_CAUSAL = 1000
N_FEATS  = 20    # total features; only 3 are truly predictive
TRUE_COEF  = {"x1": 2.0, "x2": 1.5, "x3": 0.5}
TRUE_TREAT = 2.0


def make_prediction_data(seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    cols = {}
    y    = np.zeros(N_PRED)
    for j in range(1, N_FEATS + 1):
        xj = rng.standard_normal(N_PRED)
        cols[f"x{j}"] = xj
        if j <= 3:
            y += TRUE_COEF[f"x{j}"] * xj
    y += rng.normal(0, 1.0, N_PRED)
    cols["y"] = y
    return pd.DataFrame(cols)


def make_causal_data(seed: int = 0) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    confound = rng.standard_normal(N_CAUSAL)
    treat    = 0.7 * confound + rng.normal(0, 0.8, N_CAUSAL)
    mediator = 0.6 * treat    + rng.normal(0, 0.5, N_CAUSAL)
    y        = TRUE_TREAT * treat + 3.0 * confound + rng.normal(0, 1.0, N_CAUSAL)
    return pd.DataFrame({"y": y, "treat": treat, "confound": confound, "mediator": mediator})


# ==============================================================
# 1. The Core Distinction
# ==============================================================
# Prediction asks: given x, what is the best guess for y?
#   Goal: minimize E[(ŷ - y)²] out of sample.
#   Including any correlated variable helps, even non-causal ones.
#
# Causal inference asks: if we change x by 1 unit, how does y change?
#   Goal: recover E[y(x+1) - y(x)] via a credible identification strategy.
#   Need to close backdoor paths (confounders), not just reduce residuals.
#   Adding correlated-but-non-causal predictors can bias the estimate.
#
# The two goals sometimes agree (a well-specified causal model predicts well)
# but often conflict (the best predictive model may not identify any effect).

def demo_core_distinction() -> None:
    df = make_causal_data()

    naive    = smf.ols("y ~ treat",                       data=df).fit()
    causal   = smf.ols("y ~ treat + confound",            data=df).fit()
    mediator = smf.ols("y ~ treat + confound + mediator", data=df).fit()

    print(f"\n  True total effect of treat on y: {TRUE_TREAT:.2f}")
    print(f"  Corr(treat, confound) = {df['treat'].corr(df['confound']):.3f}")
    print()
    print(f"  {'Model':>32} | {'b_treat':>8} | {'R²':>7} | In-sample MSE")
    print(f"  {'-'*32}-+-{'-'*8}-+-{'-'*7}-+-{'-'*14}")

    for label, m in [("Naive (no confound)",           naive),
                     ("Causal (add confound)",         causal),
                     ("Add mediator (bad control)",    mediator)]:
        b   = m.params["treat"]
        mse = (m.resid ** 2).mean()
        print(f"  {label:>32} | {b:>8.4f} | {m.rsquared:>7.4f} | {mse:>14.4f}")

    print()
    print("  Prediction goal  -> lowest in-sample MSE -> add confound AND mediator.")
    print("  Causal goal      -> unbiased b_treat     -> add confound, NOT mediator.")
    print()
    print("  Adding the mediator improves prediction (lower MSE, higher R²)")
    print("  but absorbs part of treat's causal path and biases the estimate down.")
    print("  Optimizing prediction and identifying causation require different choices.")


# ==============================================================
# 2. The Bias-Variance Tradeoff
# ==============================================================
# Any model makes two types of errors:
#
#   Bias²:    systematic error from underfitting (model too simple).
#   Variance: sampling error from overfitting (model too complex;
#             memorizes training noise instead of learning the signal).
#
# Test MSE = Bias² + Variance + Irreducible noise
#
# As model complexity increases:
#   - Training MSE falls monotonically.
#   - Test MSE forms a U-shape: decreases until complexity matches the
#     true function, then rises as the model memorizes training noise.

N_BV     = 300
TRUE_DEG = 2


def make_poly_data(n: int = N_BV, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-3, 3, n)
    y   = 1.0 + 0.5 * x + 0.8 * x ** 2 + rng.normal(0, 2.0, n)
    return pd.DataFrame({"y": y, "x": x})


def poly_mse(train: pd.DataFrame, test: pd.DataFrame, degree: int) -> tuple:
    """Fit a polynomial of given degree; return (train_mse, test_mse)."""
    extra = {f"x{d}": train["x"].values ** d for d in range(2, degree + 1)}
    train_aug = train.assign(**extra)
    test_aug  = test.assign(**{f"x{d}": test["x"].values ** d
                               for d in range(2, degree + 1)})
    formula = "y ~ " + " + ".join(["x"] + [f"x{d}" for d in range(2, degree + 1)])
    m = smf.ols(formula, data=train_aug).fit()
    return (m.resid ** 2).mean(), ((test_aug["y"] - m.predict(test_aug)) ** 2).mean()


def demo_bias_variance() -> None:
    df_all = make_poly_data(n=N_BV)
    split  = int(0.6 * N_BV)
    train  = df_all.iloc[:split].reset_index(drop=True)
    test   = df_all.iloc[split:].reset_index(drop=True)

    print(f"\n  Polynomial regression on quadratic DGP (true degree = {TRUE_DEG})")
    print(f"  Train N = {len(train)}, Test N = {len(test)}")
    print()
    print(f"  {'Degree':>7} | {'Train MSE':>10} | {'Test MSE':>10} | Verdict")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*20}")

    for deg in [1, 2, 3, 5, 8, 12]:
        tr_mse, te_mse = poly_mse(train, test, deg)
        if deg < TRUE_DEG:
            verdict = "underfit (high bias)"
        elif deg == TRUE_DEG:
            verdict = "correct complexity"
        elif deg <= 4:
            verdict = "slight overfit"
        else:
            verdict = "overfit (high variance)"
        print(f"  {deg:>7} | {tr_mse:>10.4f} | {te_mse:>10.4f} | {verdict}")

    print()
    print("  Train MSE falls with every added degree (more flexibility).")
    print("  Test MSE is minimized near the true degree; higher degrees")
    print("  memorize noise and generalize poorly to new data.")

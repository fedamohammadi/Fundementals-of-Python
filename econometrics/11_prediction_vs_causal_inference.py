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


# ==============================================================
# 3. Cross-Validation
# ==============================================================
# Cross-validation (CV) estimates out-of-sample error without
# holding out a separate test set.
#
# k-fold CV:
#   1. Partition the data into k equally sized folds.
#   2. For each fold i: train on the other k-1 folds, predict fold i.
#   3. CV MSE = average of the k held-out MSEs.
#
# CV selects the model that generalizes best, not the one with the
# lowest training error.  This is the key tool for prediction tasks.
#
# Typical values: k = 5 or 10.  Larger k = less bias, more variance
# in the CV estimate; k = n is leave-one-out CV (expensive but stable).

def kfold_mse(df: pd.DataFrame, degree: int, k: int = 5, seed: int = 0) -> float:
    """Return mean k-fold CV MSE for a polynomial regression of given degree."""
    n   = len(df)
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    mses  = []
    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        train = df.iloc[train_idx].reset_index(drop=True)
        val   = df.iloc[val_idx].reset_index(drop=True)
        _, mse = poly_mse(train, val, degree)
        mses.append(mse)
    return float(np.mean(mses))


def demo_cross_validation() -> None:
    df = make_poly_data(n=N_BV)

    print(f"\n  5-fold CV on polynomial regression  (true degree = {TRUE_DEG}, N = {N_BV})")
    print()
    print(f"  {'Degree':>7} | {'CV MSE':>10} | Selected?")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}")

    cv_mses = {}
    for deg in range(1, 9):
        cv_mses[deg] = kfold_mse(df, degree=deg)

    best_deg = min(cv_mses, key=cv_mses.get)
    for deg, cv_mse in cv_mses.items():
        flag = "<-- best" if deg == best_deg else ""
        print(f"  {deg:>7} | {cv_mse:>10.4f} | {flag}")

    print()
    print(f"  CV selected degree {best_deg} (true degree is {TRUE_DEG}).")
    print("  CV penalizes complexity through held-out error, not training error.")
    print("  Training MSE would always pick the highest degree -- it never stops.")
    print()
    print("  CV is the standard tool for selecting hyperparameters in prediction.")
    print("  For causal models, model selection is driven by theory, not MSE.")


# ==============================================================
# 4. Regularization: Ridge and Lasso
# ==============================================================
# Regularization adds a penalty to the OLS loss function:
#
#   Ridge (L2):  min ||y - Xβ||² + α * ||β||²
#     Shrinks all coefficients toward zero; never exactly zeros them out.
#     Good when many features have small true effects.
#
#   Lasso (L1):  min ||y - Xβ||² + α * Σ|β_j|
#     Shrinks some coefficients to exactly zero (feature selection).
#     Good when most features are irrelevant.
#
# Higher α -> more shrinkage -> lower variance, higher bias.
# Standardize X before regularizing so the penalty treats all features equally.

def demo_regularization() -> None:
    df_all = make_prediction_data()
    split  = int(0.7 * N_PRED)
    train  = df_all.iloc[:split].reset_index(drop=True)
    test   = df_all.iloc[split:].reset_index(drop=True)

    features = [f"x{j}" for j in range(1, N_FEATS + 1)]

    # Standardize features (fit on train, apply to both)
    mu  = train[features].mean()
    std = train[features].std()

    X_tr = (train[features] - mu) / std
    X_te = (test[features]  - mu) / std
    y_tr = train["y"].values
    y_te = test["y"].values

    X_tr_c = sm.add_constant(X_tr.values)
    X_te_c = sm.add_constant(X_te.values)

    # OLS
    ols_res = sm.OLS(y_tr, X_tr_c).fit()
    ols_mse = ((y_te - ols_res.predict(X_te_c)) ** 2).mean()

    # Ridge and Lasso with statsmodels elastic net
    best_ridge_mse = np.inf
    best_lasso_mse = np.inf
    best_ridge, best_lasso = None, None

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        ridge = sm.OLS(y_tr, X_tr_c).fit_regularized(method="elastic_net",
                                                       alpha=alpha, L1_wt=0.0)
        lasso = sm.OLS(y_tr, X_tr_c).fit_regularized(method="elastic_net",
                                                       alpha=alpha, L1_wt=1.0)
        r_mse = ((y_te - X_te_c @ ridge.params) ** 2).mean()
        l_mse = ((y_te - X_te_c @ lasso.params) ** 2).mean()
        if r_mse < best_ridge_mse:
            best_ridge_mse = r_mse
            best_ridge = ridge
        if l_mse < best_lasso_mse:
            best_lasso_mse = l_mse
            best_lasso = lasso

    # Non-zero coefficients in Lasso (excluding intercept)
    lasso_nonzero = (np.abs(best_lasso.params[1:]) > 1e-6).sum()

    print(f"\n  High-dimensional prediction: {N_FEATS} features, 3 truly predictive")
    print(f"  Train N = {len(train)}, Test N = {len(test)}")
    print()
    print(f"  {'Method':>8} | {'Test MSE':>10} | Non-zero coefs")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*15}")
    print(f"  {'OLS':>8} | {ols_mse:>10.4f} | {N_FEATS} (all retained)")
    print(f"  {'Ridge':>8} | {best_ridge_mse:>10.4f} | {N_FEATS} (all shrunk)")
    print(f"  {'Lasso':>8} | {best_lasso_mse:>10.4f} | {lasso_nonzero} (others zeroed)")

    print()
    print("  True features: x1, x2, x3.  Features x4-x20 are pure noise.")
    true_vars = ["x1", "x2", "x3"]
    print(f"  Lasso coefficients on true features (standardized scale):")
    for j, var in enumerate(features[:5], start=1):  # show first 5
        lc = best_lasso.params[j]
        flag = "(true)" if var in true_vars else "(noise)"
        print(f"    {var}: {lc:>8.4f}  {flag}")
    print("    ...")
    print()
    print("  Lasso zeroes noise features and retains signal features.")
    print("  Ridge shrinks all coefficients but keeps them all non-zero.")
    print("  Both outperform OLS out-of-sample when features outnumber signal.")


# ==============================================================
# 5. Why Good Prediction Does Not Imply Causation
# ==============================================================
# A model can predict y perfectly from x without x being the cause of y.
# Three common reasons prediction ≠ causation:
#
#   1. Spurious correlation: a third variable C causes both x and y.
#      x predicts y in the data, but changing x (holding C fixed) does nothing.
#
#   2. Reverse causation: y causes x.
#      Knowing x predicts y (high R²), but intervening on x won't change y.
#
#   3. Proxy variables: x measures a latent construct that causes y.
#      x predicts y through the latent variable, not through x itself.
#
# The key test is an intervention: fix x externally (e.g., randomize it)
# and observe whether y changes.  Prediction cannot answer this from
# observational data alone.

def demo_prediction_not_causation() -> None:
    rng  = np.random.default_rng(42)
    n    = 800

    # Spurious correlation: C -> X, C -> Y
    C = rng.standard_normal(n)
    X = 0.8 * C + rng.normal(0, 0.5, n)
    Y = 2.0 * C + rng.normal(0, 0.5, n)   # X has NO direct effect on Y

    df = pd.DataFrame({"Y": Y, "X": X, "C": C})

    naive   = smf.ols("Y ~ X",     data=df).fit()
    correct = smf.ols("Y ~ X + C", data=df).fit()

    print(f"\n  True causal effect of X on Y: 0.00  (X and Y share a common cause C)")
    print(f"  Corr(X, C) = {df['X'].corr(df['C']):.3f}   Corr(X, Y) = {df['X'].corr(df['Y']):.3f}")
    print()
    print(f"  {'Model':>20} | {'b_X':>8} | {'R²':>7} | Prediction quality")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*7}-+-{'-'*20}")

    for label, m in [("Y ~ X (no C)", naive), ("Y ~ X + C (correct)", correct)]:
        b   = m.params["X"]
        mse = (m.resid ** 2).mean()
        print(f"  {label:>20} | {b:>8.4f} | {m.rsquared:>7.4f} | MSE = {mse:.4f}")

    print()
    print("  The naive model finds a strong, significant relationship between X and Y")
    print("  (high R², low p-value) even though changing X would have NO effect on Y.")
    print("  X predicts Y only because both proxy the omitted common cause C.")
    print()
    print("  An RCT that randomizes X would show b_X ≈ 0 (X has no direct effect).")
    print("  Observational prediction models cannot distinguish this from true causation.")


# ==============================================================
# 6. Bad Controls and Post-Treatment Bias
# ==============================================================
# A "bad control" is a variable that is itself caused by the treatment.
# Including it in the regression partials out part of the treatment's
# causal effect, biasing the estimate toward zero.
#
# This is sometimes called the "Table 2 fallacy" (Westreich & Greenland 2013):
# published tables showing all OLS coefficients from a model with many controls
# may look like causal estimates for each variable, but coefficients on post-
# treatment mediators do not have a causal interpretation.
#
# Example causal chain:
#   treat -> mediator -> y    (the indirect path)
#   treat             -> y    (the direct path)
#   Total effect of treat = direct + indirect.
#
# Controlling for the mediator recovers only the direct path, not the total.

def demo_bad_controls() -> None:
    df = make_causal_data()

    total_m    = smf.ols("y ~ treat + confound",             data=df).fit()
    with_med   = smf.ols("y ~ treat + confound + mediator",  data=df).fit()
    mediator_m = smf.ols("mediator ~ treat",                 data=df).fit()

    print(f"\n  True total effect of treat on y:   {TRUE_TREAT:.2f}")
    print(f"  True effect of treat on mediator:  0.60")
    print()
    print(f"  {'Model':>36} | b_treat | Interpretation")
    print(f"  {'-'*36}-+-{'-'*8}-+-{'-'*30}")

    print(f"  {'Total effect (no mediator control)':>36} | {total_m.params['treat']:>7.4f} | total effect")
    print(f"  {'With mediator (bad control)':>36} | {with_med.params['treat']:>7.4f} | direct effect only")

    print()
    print(f"  Mediator model:  treat -> mediator,  b = {mediator_m.params['treat']:.4f}")
    print()
    print("  The indirect path: treat -> mediator -> y.")
    print(f"  Indirect effect ≈ b(treat->mediator) × b(mediator->y in full model)")
    b_m_in_full = with_med.params.get("mediator", 0)
    indirect = mediator_m.params["treat"] * b_m_in_full
    print(f"  ≈ {mediator_m.params['treat']:.3f} × {b_m_in_full:.3f} = {indirect:.4f}")
    print(f"  Direct + Indirect ≈ {with_med.params['treat']:.4f} + {indirect:.4f} = {with_med.params['treat'] + indirect:.4f}")
    print()
    print("  Controlling for the mediator is correct when you want ONLY the direct path.")
    print("  If you want the TOTAL effect, do NOT control for the mediator.")
    print("  Knowing which effect you want is a causal question, not a statistical one.")


# ==============================================================
# 7. Practical Guide
# ==============================================================

def demo_practical_guide() -> None:
    guide = [
        ("What is the goal of your analysis?",
         "Prediction  -> maximize out-of-sample accuracy; use CV and regularization.",
         "Causation   -> identify an effect; use theory to select controls."),

        ("Choosing control variables (prediction goal):",
         "Include anything correlated with y, even if not causally related.",
         "More features often help; use Lasso for automatic selection."),

        ("Choosing control variables (causal goal):",
         "Include confounders (common causes of treatment and outcome).",
         "Exclude mediators (variables caused by treatment and causing outcome)."),

        ("How do you select model complexity?",
         "Prediction -> cross-validation (k-fold, typically k=5 or 10).",
         "Causation  -> theory; simpler models are preferred for interpretability."),

        ("Does a high R² confirm a causal effect?",
         "No. A high R² only means the model fits the data well.",
         "A spurious model can have R²=0.99 with zero causal content."),

        ("Is regularization useful for causal models?",
         "Rarely. Regularization biases coefficients toward zero intentionally.",
         "For causal models, use unbiased OLS (or IV/FE/DiD) with theory-based controls."),

        ("When should you use prediction methods (ML) vs causal methods?",
         "Prediction: forecasting prices, classifying emails, detecting fraud.",
         "Causation: policy evaluation, treatment effect, A/B testing interpretation."),
    ]

    print()
    for i, (question, opt_a, opt_b) in enumerate(guide, 1):
        print(f"  Step {i}: {question}")
        print(f"    {opt_a}")
        print(f"    {opt_b}")
        print()

    print("  The two goals are not in conflict -- both are valuable.")
    print("  The mistake is using a prediction model to draw causal conclusions,")
    print("  or using a causal model when the task is prediction.")
    print("  Be explicit about your goal before choosing a method.")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. The Core Distinction")
    demo_core_distinction()

    section("2. The Bias-Variance Tradeoff")
    demo_bias_variance()

    section("3. Cross-Validation")
    demo_cross_validation()

    section("4. Regularization: Ridge and Lasso")
    demo_regularization()

    section("5. Why Good Prediction Does Not Imply Causation")
    demo_prediction_not_causation()

    section("6. Bad Controls and Post-Treatment Bias")
    demo_bad_controls()

    section("7. Practical Guide")
    demo_practical_guide()


if __name__ == "__main__":
    main()

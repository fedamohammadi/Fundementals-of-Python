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


# ==============================================================
# 3. Logit: Logistic Regression
# ==============================================================
# The logit model specifies:
#
#   P(y=1 | x) = Λ(xβ) = 1 / (1 + exp(-xβ))
#
# The log-odds (logit link) is linear in x:
#
#   log[ P / (1-P) ] = xβ
#
# β_j is the change in log-odds per unit increase in x_j.
# The odds ratio exp(β_j) > 1 means higher x_j raises P(y=1).
#
# Estimated by maximum likelihood.  In large samples, MLE is unbiased,
# consistent, and asymptotically normal (z-tests, not t-tests).

def demo_logit() -> None:
    df    = make_binary_data()
    model = smf.logit("participate ~ educ + exper + exper2 + kids", data=df).fit(disp=False)

    print(f"\n  Logit: log-odds coefficients  (N = {N_OBS})")
    print()
    print(f"  {'Variable':>10} | {'Coef':>8} | {'SE':>7} | {'z':>7} | {'p':>8} | Odds Ratio")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*11}")

    for var in ["Intercept", "educ", "exper", "exper2", "kids"]:
        b  = model.params[var]
        se = model.bse[var]
        z  = model.tvalues[var]
        p  = model.pvalues[var]
        or_ = np.exp(b)
        print(f"  {var:>10} | {b:>8.4f} | {se:>7.4f} | {z:>7.2f} | {p:>8.4f} | {or_:>11.4f}")

    print()
    print("  True DGP: educ=0.08, exper=0.05, exper²=-0.001, kids=-0.80")
    print("  (Logit is the true model here; estimates should be close.)")
    print()
    or_educ = np.exp(model.params["educ"])
    print(f"  Odds ratio for educ = {or_educ:.4f}")
    print("  Each additional year of education multiplies the participation")
    print(f"  odds by {or_educ:.3f} -- a {(or_educ - 1)*100:.1f}% increase in the odds.")


# ==============================================================
# 4. Probit: Normal CDF Model
# ==============================================================
# The probit model specifies:
#
#   P(y=1 | x) = Φ(xβ)
#
# Φ is the standard normal CDF.  Coefficients are on the scale of
# standard deviations of the latent index.
#
# Logit vs probit in practice:
#   - Raw coefficients differ by ~1.81 (the π/√3 scale factor).
#   - Predicted probabilities are nearly identical.
#   - Logit is more common in economics (log-odds are convenient).
#   - Choice between them almost never affects substantive conclusions.

def demo_probit() -> None:
    df         = make_binary_data()
    logit_res  = smf.logit( "participate ~ educ + exper + exper2 + kids", data=df).fit(disp=False)
    probit_res = smf.probit("participate ~ educ + exper + exper2 + kids", data=df).fit(disp=False)

    scale = np.pi / np.sqrt(3)

    print(f"\n  Logit vs Probit  (scale factor π/√3 ≈ {scale:.4f})")
    print()
    print(f"  {'Variable':>10} | {'Logit β':>9} | {'Probit β':>9} | {'Ratio':>7} | Rescaled probit")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*16}")

    for var in ["educ", "exper", "exper2", "kids"]:
        bl      = logit_res.params[var]
        bp      = probit_res.params[var]
        ratio   = bl / bp if abs(bp) > 1e-8 else float("nan")
        rescaled = bp * scale
        print(f"  {var:>10} | {bl:>9.4f} | {bp:>9.4f} | {ratio:>7.3f} | {rescaled:>16.4f}")

    p_logit  = logit_res.predict()
    p_probit = probit_res.predict()
    corr     = np.corrcoef(p_logit, p_probit)[0, 1]
    mae      = np.abs(p_logit - p_probit).mean()

    print()
    print(f"  Predicted probability agreement:")
    print(f"    Correlation:               {corr:.6f}")
    print(f"    Mean absolute difference:  {mae:.6f}")
    print()
    print("  Despite different raw coefficients, both models produce")
    print("  nearly identical predicted probabilities.  Report the model")
    print("  conventional in your field; add the other as a robustness check.")


# ==============================================================
# 5. Marginal Effects
# ==============================================================
# In a linear model, β directly measures ∂E[y]/∂x_j.
# In logit/probit, the marginal effect on P(y=1) is:
#
#   ∂P/∂x_j = f(xβ) × β_j
#
# where f is the PDF of the model (logistic or normal).
# This varies across observations, so we summarize it:
#
#   AME (Average Marginal Effect): average of individual f(xᵢβ)×β_j.
#        Recommended for most reports; comparable to the LPM coefficient.
#
#   MEM (Marginal Effect at the Mean): f(x̄β)×β_j, evaluated at the
#        sample mean of x.  Simpler but can be misleading if x̄ is not
#        representative (e.g., continuous x with a bimodal distribution).
#
# For binary x_j: the marginal effect is ΔP = Λ(xβ + β_j) - Λ(xβ).

def demo_marginal_effects() -> None:
    df    = make_binary_data()
    model = smf.logit("participate ~ educ + exper + exper2 + kids", data=df).fit(disp=False)

    ame = model.get_margeff(at="overall")
    mem = model.get_margeff(at="mean")

    ame_effects = dict(zip(ame.summary_frame().index, ame.margeff))
    ame_ses     = dict(zip(ame.summary_frame().index, ame.margeff_se))
    mem_effects = dict(zip(mem.summary_frame().index, mem.margeff))
    mem_ses     = dict(zip(mem.summary_frame().index, mem.margeff_se))

    print(f"\n  Marginal effects on P(participate=1)")
    print()
    print(f"  {'Variable':>10} | {'AME':>9} | {'AME SE':>8} | {'MEM':>9} | MEM SE")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

    for var in ["educ", "exper", "exper2", "kids"]:
        print(f"  {var:>10} | {ame_effects[var]:>9.4f} | {ame_ses[var]:>8.4f} | "
              f"{mem_effects[var]:>9.4f} | {mem_ses[var]:>8.4f}")

    print()
    print(f"  AME for educ: {ame_effects['educ']:.4f}")
    print("  One extra year of education raises P(participate) by")
    print(f"  {ame_effects['educ']:.4f} on average (averaged over all observations).")
    print()
    print("  True DGP: educ coefficient = 0.08 in log-odds.")
    print("  AME converts this to a probability scale; the exact value")
    print("  depends on the distribution of xβ in the sample.")
    print()
    print("  AME is preferred over MEM: it averages over the actual")
    print("  sample distribution rather than a hypothetical average person.")


# ==============================================================
# 6. Model Fit and Diagnostics
# ==============================================================
# Standard R² doesn't apply to binary outcomes.  Common alternatives:
#
#   McFadden pseudo-R²:  1 - LL_full / LL_null
#     Ranges [0, 1]; values of 0.2–0.4 indicate good fit.
#
#   AIC = -2*LL + 2k      BIC = -2*LL + k*ln(n)
#     Lower is better.  Compare nested and non-nested models.
#
#   Accuracy: fraction correctly classified at a given threshold.
#
#   Brier score: E[(p̂ - y)²].  Lower is better; 0.25 = naive baseline.

def demo_model_fit() -> None:
    df   = make_binary_data()
    null = smf.logit("participate ~ 1",    data=df).fit(disp=False)
    red  = smf.logit("participate ~ educ", data=df).fit(disp=False)
    full = smf.logit("participate ~ educ + exper + exper2 + kids", data=df).fit(disp=False)

    print(f"\n  {'Model':>24} | {'McF-R²':>8} | {'AIC':>9} | {'BIC':>9} | {'Accuracy':>9} | Brier")
    print(f"  {'-'*24}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")

    for label, m in [("Null (intercept only)", null),
                     ("Reduced (educ only)",   red),
                     ("Full model",            full)]:
        phat   = m.predict()
        mcf    = 1 - m.llf / null.llf
        acc    = ((phat >= 0.5).astype(int) == df["participate"]).mean()
        brier  = ((phat - df["participate"]) ** 2).mean()
        print(f"  {label:>24} | {mcf:>8.4f} | {m.aic:>9.2f} | {m.bic:>9.2f} | "
              f"{acc:>9.4f} | {brier:.4f}")

    print()
    print("  McFadden R² > 0.2 indicates a reasonably good fit.")
    print("  AIC/BIC both prefer the full model (lower values).")
    print("  Accuracy improves but can be misleading with imbalanced outcomes;")
    print("  always pair it with Brier score or AUC.")


# ==============================================================
# 7. Predicted Probabilities and Threshold Selection
# ==============================================================
# The default 0.5 threshold maximizes accuracy when classes are balanced,
# but the optimal threshold depends on the cost of FP vs FN.
#
# Key metrics as a function of threshold:
#   Precision = TP / (TP + FP)   -- of those predicted positive, how many are?
#   Recall    = TP / (TP + FN)   -- of all positives, how many did we catch?
#   F1        = 2 * P*R / (P+R)  -- harmonic mean of precision and recall.
#
# AUC (area under ROC curve):
#   Summary of the model's discriminating power across all thresholds.
#   AUC = 0.5 is random chance; AUC = 1.0 is perfect discrimination.

def demo_predicted_probs() -> None:
    df   = make_binary_data()
    fit  = smf.logit("participate ~ educ + exper + exper2 + kids", data=df).fit(disp=False)
    phat = fit.predict()
    y    = df["participate"].values

    print(f"\n  Predicted probability summary:")
    print(f"    Mean: {phat.mean():.3f}   Std: {phat.std():.3f}"
          f"   Min: {phat.min():.3f}   Max: {phat.max():.3f}")
    print()
    print(f"  {'Threshold':>10} | {'Accuracy':>9} | {'Precision':>10} | {'Recall':>8} | F1")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        yhat = (phat >= thresh).astype(int)
        tp   = int(((yhat == 1) & (y == 1)).sum())
        fp   = int(((yhat == 1) & (y == 0)).sum())
        fn   = int(((yhat == 0) & (y == 1)).sum())
        tn   = int(((yhat == 0) & (y == 0)).sum())
        acc  = (tp + tn) / len(y)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {thresh:>10.1f} | {acc:>9.4f} | {prec:>10.4f} | {rec:>8.4f} | {f1:.4f}")

    # AUC via trapezoidal rule over 200 threshold steps
    thresholds = np.linspace(0, 1, 201)
    tprs, fprs = [], []
    for t in thresholds:
        yhat = (phat >= t).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    pairs        = sorted(zip(fprs, tprs))
    fprs_sorted, tprs_sorted = zip(*pairs)
    auc = float(np.trapz(tprs_sorted, fprs_sorted))

    print()
    print(f"  AUC = {auc:.4f}  (0.5 = random, 1.0 = perfect)")
    print("  AUC > 0.7 is acceptable; > 0.8 is good; > 0.9 is excellent.")
    print()
    print("  Lower thresholds increase recall (catch more positives) at the")
    print("  cost of precision (more false positives).  Choose based on costs.")

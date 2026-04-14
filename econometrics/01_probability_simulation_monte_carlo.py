"""
Probability, Simulation, and Monte Carlo Methods
=================================================
Concepts covered:
- Random variables and common probability distributions
- Expected value and variance (formula vs. simulation)
- Law of Large Numbers
- Central Limit Theorem
- Monte Carlo integration
- Monte Carlo simulation for a practical economic problem

Why this matters in econometrics:
  Econometrics is built on probability theory. Before running a regression
  you need to understand what a random variable is, what a distribution
  looks like, and how large samples behave. Monte Carlo methods let you
  *verify* analytical results by brute-force simulation — an indispensable
  sanity check when the math gets complicated.
"""

import random
import math


# ==============================================================
# Helper: print a clean section header
# ==============================================================

def section(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# 1. Random Variables and Probability Distributions
# ==============================================================
# A random variable maps outcomes of a random process to numbers.
# Each distribution has parameters that control its shape.
# We simulate draws using Python's built-in `random` module
# (numpy.random is introduced later; pure stdlib keeps this
#  accessible to anyone who just installed Python).

def demo_distributions(n: int = 10, seed: int = 42) -> None:
    """
    Demonstrate five fundamental distributions by drawing n samples
    and printing them alongside their key parameters.

    Parameters
    ----------
    n    : number of samples to draw per distribution
    seed : random seed for reproducibility
    """
    random.seed(seed)

    # ----------------------------------------------------------
    # Uniform distribution  U(a, b)
    # Every value between a and b is equally likely.
    # Mean = (a + b) / 2,  Variance = (b - a)^2 / 12
    # Use-case: prior belief with no information, random assignment.
    # ----------------------------------------------------------
    a, b = 0.0, 10.0
    uniform_draws = [random.uniform(a, b) for _ in range(n)]
    print(f"\n[Uniform U({a}, {b})]")
    print(f"  Theoretical mean     : {(a + b) / 2:.2f}")
    print(f"  Theoretical variance : {(b - a)**2 / 12:.2f}")
    print(f"  Sample ({n} draws)  : {[round(x, 2) for x in uniform_draws]}")

    # ----------------------------------------------------------
    # Normal distribution  N(mu, sigma)
    # The bell curve. Most continuous outcome variables in
    # econometrics are approximately normal (or assumed to be).
    # Mean = mu,  Variance = sigma^2
    # ----------------------------------------------------------
    mu, sigma = 50_000, 15_000   # e.g. annual income in USD
    normal_draws = [random.gauss(mu, sigma) for _ in range(n)]
    print(f"\n[Normal N(mu={mu:,}, sigma={sigma:,})]")
    print(f"  Theoretical mean     : {mu:,.0f}")
    print(f"  Theoretical variance : {sigma**2:,.0f}")
    print(f"  Sample ({n} draws)  : {[f'{x:,.0f}' for x in normal_draws]}")

    # ----------------------------------------------------------
    # Bernoulli distribution  Bern(p)
    # A single coin-flip: 1 (success) with probability p,
    #                     0 (failure) with probability 1-p.
    # Mean = p,  Variance = p*(1-p)
    # Use-case: employment status, treatment assignment.
    # ----------------------------------------------------------
    p = 0.3    # e.g. probability of being unemployed
    bernoulli_draws = [1 if random.random() < p else 0 for _ in range(n)]
    print(f"\n[Bernoulli Bern(p={p})]")
    print(f"  Theoretical mean     : {p:.2f}")
    print(f"  Theoretical variance : {p*(1-p):.4f}")
    print(f"  Sample ({n} draws)  : {bernoulli_draws}")

    # ----------------------------------------------------------
    # Binomial distribution  Bin(n_trials, p)
    # Number of successes in n_trials independent Bernoulli trials.
    # Mean = n_trials * p,  Variance = n_trials * p * (1-p)
    # Use-case: number of job offers out of k applications.
    # ----------------------------------------------------------
    n_trials, p_binom = 20, 0.4
    binom_draws = [sum(1 if random.random() < p_binom else 0
                       for _ in range(n_trials))
                   for _ in range(n)]
    print(f"\n[Binomial Bin(n={n_trials}, p={p_binom})]")
    print(f"  Theoretical mean     : {n_trials * p_binom:.2f}")
    print(f"  Theoretical variance : {n_trials * p_binom * (1-p_binom):.4f}")
    print(f"  Sample ({n} draws)  : {binom_draws}")

    # ----------------------------------------------------------
    # Exponential distribution  Exp(lambda)
    # Models waiting time between events in a Poisson process.
    # Mean = 1/lambda,  Variance = 1/lambda^2
    # Use-case: duration of unemployment spells, time between defaults.
    # ----------------------------------------------------------
    lam = 0.5   # rate = 0.5 events per unit time → mean wait = 2
    exp_draws = [random.expovariate(lam) for _ in range(n)]
    print(f"\n[Exponential Exp(lambda={lam})]")
    print(f"  Theoretical mean     : {1/lam:.2f}")
    print(f"  Theoretical variance : {1/lam**2:.2f}")
    print(f"  Sample ({n} draws)  : {[round(x, 2) for x in exp_draws]}")


# ==============================================================
# 2. Expected Value and Variance — Formula vs. Simulation
# ==============================================================
# The expected value E[X] is the long-run average outcome.
# Variance Var(X) = E[(X - E[X])^2] measures spread around the mean.
#
# Key insight: simulation *converges* to the true value as n → ∞.
# Showing this convergence demystifies probability theory — the
# formulas are just shortcuts for what the simulation does slowly.

def sample_mean(data: list[float]) -> float:
    """Return the arithmetic mean of a list."""
    return sum(data) / len(data)


def sample_variance(data: list[float], ddof: int = 1) -> float:
    """
    Return the sample variance.

    Parameters
    ----------
    ddof : degrees of freedom correction.
           Use ddof=1 (Bessel's correction) for unbiased estimate.
           Use ddof=0 for the population formula.
    """
    m = sample_mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def demo_ev_variance(seed: int = 42) -> None:
    """
    Compare theoretical E[X] and Var(X) with simulation estimates
    for a Normal distribution, across increasing sample sizes.

    This is the core idea behind simulation-based inference:
    as n grows, the sample statistics approach the truth.
    """
    random.seed(seed)

    mu, sigma = 0.0, 1.0          # standard normal N(0, 1)
    true_mean = mu
    true_var  = sigma ** 2

    print(f"\n  True mean     = {true_mean:.4f}")
    print(f"  True variance = {true_var:.4f}\n")
    print(f"  {'n':>10} | {'sim. mean':>12} | {'sim. var':>12} | "
          f"{'mean error':>12} | {'var error':>12}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for n in [10, 100, 1_000, 10_000, 100_000]:
        draws = [random.gauss(mu, sigma) for _ in range(n)]
        sim_mean = sample_mean(draws)
        sim_var  = sample_variance(draws, ddof=1)

        mean_err = abs(sim_mean - true_mean)
        var_err  = abs(sim_var  - true_var)

        print(f"  {n:>10,} | {sim_mean:>12.5f} | {sim_var:>12.5f} | "
              f"{mean_err:>12.5f} | {var_err:>12.5f}")

    print()
    print("  Observation: both errors shrink toward zero as n grows.")
    print("  This is the Law of Large Numbers in action (see Section 3).")


# ==============================================================
# 3. Law of Large Numbers (LLN)
# ==============================================================
# The LLN states that the sample average of iid random variables
# converges to the true expected value as the sample size grows.
#
# Formal statement (Weak LLN):
#   If X_1, X_2, ..., X_n are iid with mean mu, then
#   (X_1 + ... + X_n) / n  →  mu  as  n → ∞
#
# Why it matters in econometrics:
#   - Justifies using sample moments to estimate population moments.
#   - Underpins OLS consistency: with enough data, OLS converges
#     to the true population coefficients.

def demo_lln(seed: int = 42) -> None:
    """
    Visualize the LLN by tracking the running mean of coin flips.

    A fair coin has E[X] = 0.5 (1 = heads, 0 = tails).
    We show how the running average approaches 0.5 step by step.
    """
    random.seed(seed)

    n_total = 1000
    flips   = [1 if random.random() < 0.5 else 0 for _ in range(n_total)]

    # Track the running (cumulative) mean after each flip
    running_sum  = 0
    checkpoints  = [1, 5, 10, 50, 100, 250, 500, 1000]
    print(f"\n  True expected value (fair coin) = 0.5\n")
    print(f"  {'n':>6} | {'running mean':>14} | {'error from 0.5':>16}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*16}")

    for i, flip in enumerate(flips, start=1):
        running_sum += flip
        running_mean = running_sum / i
        if i in checkpoints:
            error = abs(running_mean - 0.5)
            print(f"  {i:>6,} | {running_mean:>14.5f} | {error:>16.5f}")

    print()
    print("  The running mean zigzags early but locks in near 0.5 by n=1,000.")
    print("  With n=1,000,000 it would be indistinguishable from 0.5.")


# ==============================================================
# 4. Central Limit Theorem (CLT)
# ==============================================================
# The CLT is arguably the most important theorem in statistics.
# It says that the *sampling distribution of the mean* is
# approximately normal, regardless of the original distribution,
# provided n is large enough.
#
# Formal statement:
#   sqrt(n) * (X_bar - mu) / sigma  →  N(0, 1)  as n → ∞
#
# Why it matters in econometrics:
#   - Justifies normal-based inference (t-tests, confidence intervals)
#     even when the underlying error distribution is unknown.
#   - Explains why OLS residuals look approximately normal in large
#     samples even if the true error is skewed or fat-tailed.
#   - Motivates the use of asymptotic standard errors.

def demo_clt(
    n_samples: int = 5_000,
    sample_size: int = 30,
    seed: int = 42,
) -> None:
    """
    Demonstrate the CLT using a skewed Exponential distribution.

    We draw many samples of size `sample_size` from Exp(lambda=0.5),
    compute the sample mean for each, and show that those means
    are approximately normally distributed — even though the raw
    data is heavily right-skewed.

    Parameters
    ----------
    n_samples   : number of samples to draw (i.e. repetitions)
    sample_size : size of each sample (n in the CLT formula)
    seed        : random seed
    """
    random.seed(seed)

    lam = 0.5              # rate parameter
    true_mean = 1 / lam    # E[Exp(lambda)] = 1/lambda = 2.0
    true_std  = 1 / lam    # SD[Exp(lambda)] = 1/lambda = 2.0

    # Collect the sample mean from each repetition
    sample_means = []
    for _ in range(n_samples):
        sample = [random.expovariate(lam) for _ in range(sample_size)]
        sample_means.append(sample_mean(sample))

    # The CLT predicts the sampling distribution of X_bar is:
    #   N(mu, sigma^2 / n)
    #   → N(2.0, 4.0 / 30)  with SE = sigma / sqrt(n)
    predicted_se = true_std / math.sqrt(sample_size)

    # Compare predicted vs simulated statistics
    sim_mean = sample_mean(sample_means)
    sim_std  = math.sqrt(sample_variance(sample_means, ddof=1))

    print(f"\n  Source distribution : Exponential(lambda={lam})")
    print(f"  True mean (mu)      : {true_mean:.4f}")
    print(f"  Sample size (n)     : {sample_size}")
    print(f"  Number of samples   : {n_samples:,}")
    print()
    print(f"  CLT prediction for sampling distribution of X_bar:")
    print(f"    Center (mu)       = {true_mean:.4f}")
    print(f"    Std error (SE)    = sigma/sqrt(n) = {predicted_se:.4f}")
    print()
    print(f"  Simulated sampling distribution of X_bar:")
    print(f"    Simulated mean    = {sim_mean:.4f}   (should ≈ {true_mean:.4f})")
    print(f"    Simulated std     = {sim_std:.4f}   (should ≈ {predicted_se:.4f})")

    # Quick normality check: what fraction of sample means fall within
    # ±1 SE, ±2 SE, ±3 SE of the predicted center?
    # For N(0,1): ~68%, ~95%, ~99.7%
    print()
    print("  Normality check — share of sample means within k×SE of mu:")
    for k in [1, 2, 3]:
        lo = true_mean - k * predicted_se
        hi = true_mean + k * predicted_se
        inside = sum(1 for m in sample_means if lo <= m <= hi)
        pct = inside / n_samples * 100
        print(f"    ±{k} SE : {pct:.1f}%  (normal theory: "
              f"{[68.3, 95.4, 99.7][k-1]:.1f}%)")

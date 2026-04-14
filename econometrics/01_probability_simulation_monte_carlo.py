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
import sys

# Ensure unicode characters (Greek letters, math symbols) render
# correctly on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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


# ==============================================================
# 5. Monte Carlo Integration
# ==============================================================
# Monte Carlo integration estimates a definite integral by
# drawing random points and checking whether they satisfy a
# condition — or by averaging function values at random inputs.
#
# Classic formula (hit-or-miss variant):
#   ∫_a^b f(x) dx  ≈  (b - a) * (1/n) * Σ f(x_i)
#   where x_i ~ Uniform(a, b)
#
# Why useful in econometrics:
#   - Evaluate high-dimensional integrals (e.g. likelihood functions)
#     that have no closed form.
#   - Compute probabilities under non-standard distributions.
#   - Estimate option prices, expected welfare, etc.

def mc_integrate(
    func,
    a: float,
    b: float,
    n: int,
    seed: int = 42,
) -> float:
    """
    Estimate ∫_a^b func(x) dx using n random draws from U(a, b).

    Parameters
    ----------
    func : callable, the integrand
    a, b : limits of integration
    n    : number of Monte Carlo samples
    seed : random seed

    Returns
    -------
    Estimated value of the integral.
    """
    random.seed(seed)
    total = sum(func(random.uniform(a, b)) for _ in range(n))
    return (b - a) * total / n


def demo_mc_integration() -> None:
    """
    Estimate three integrals with known analytical answers and
    show how the MC estimate improves as n grows.

    Example 1: ∫_0^1 x^2 dx = 1/3 ≈ 0.3333
    Example 2: ∫_0^π sin(x) dx = 2
    Example 3: Estimate π using the unit-circle trick
               (fraction of points inside unit circle × 4)
    """
    # ----------------------------------------------------------
    # Example 1: ∫_0^1 x^2 dx  (true answer = 1/3)
    # ----------------------------------------------------------
    true_1 = 1 / 3
    print(f"\n  Integral 1: ∫_0^1 x² dx  (true = {true_1:.6f})")
    print(f"  {'n':>10} | {'estimate':>12} | {'error':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*10}")
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        est = mc_integrate(lambda x: x ** 2, 0, 1, n, seed=0)
        print(f"  {n:>10,} | {est:>12.6f} | {abs(est - true_1):>10.6f}")

    # ----------------------------------------------------------
    # Example 2: ∫_0^π sin(x) dx  (true answer = 2)
    # ----------------------------------------------------------
    true_2 = 2.0
    print(f"\n  Integral 2: ∫_0^π sin(x) dx  (true = {true_2:.6f})")
    print(f"  {'n':>10} | {'estimate':>12} | {'error':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*10}")
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        est = mc_integrate(math.sin, 0, math.pi, n, seed=1)
        print(f"  {n:>10,} | {est:>12.6f} | {abs(est - true_2):>10.6f}")

    # ----------------------------------------------------------
    # Example 3: Estimating π via the unit-circle trick
    # Throw n darts at a 2×2 square centered at the origin.
    # The fraction that land inside the unit circle × 4 ≈ π.
    # Area of circle = π r² = π  (r=1)
    # Area of square = (2r)² = 4
    # → π/4 = Prob(inside circle)  → π ≈ 4 × (hits / n)
    # ----------------------------------------------------------
    print(f"\n  Integral 3: Estimating π via random darts")
    print(f"  True π = {math.pi:.8f}")
    print(f"  {'n':>10} | {'π estimate':>12} | {'error':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*10}")
    random.seed(7)
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        hits = sum(
            1
            for _ in range(n)
            if random.uniform(-1, 1) ** 2 + random.uniform(-1, 1) ** 2 <= 1.0
        )
        pi_est = 4 * hits / n
        print(f"  {n:>10,} | {pi_est:>12.6f} | {abs(pi_est - math.pi):>10.6f}")


# ==============================================================
# 6. Monte Carlo Simulation — Practical Economic Example
# ==============================================================
# Suppose a worker has uncertain annual earnings drawn from
# N(50_000, 10_000²) and must pay a flat 20% income tax plus
# a fixed healthcare premium of $5,000/year.
#
# Question: What is the probability that the worker's *after-tax
# after-premium disposable income* falls below the poverty line
# of $20,000?
#
# Analytical derivation is possible here, but Monte Carlo makes
# the logic completely transparent and extends trivially to more
# complex, non-linear rules.

def worker_disposable_income(
    gross: float,
    tax_rate: float = 0.20,
    healthcare_premium: float = 5_000.0,
) -> float:
    """
    Return disposable income after tax and fixed healthcare premium.

    Parameters
    ----------
    gross             : gross annual earnings in USD
    tax_rate          : flat income tax rate (default 20%)
    healthcare_premium: fixed annual healthcare cost in USD

    Returns
    -------
    Disposable income in USD.
    """
    after_tax = gross * (1 - tax_rate)
    return after_tax - healthcare_premium


def demo_mc_simulation(
    n: int = 100_000,
    seed: int = 42,
) -> None:
    """
    Estimate poverty-risk probabilities for a worker with uncertain
    income using Monte Carlo simulation.

    We also show how the simulation degrades gracefully when the
    earnings distribution is changed (e.g. fat-tailed vs normal).

    Parameters
    ----------
    n    : number of simulated workers
    seed : random seed
    """
    random.seed(seed)

    mu_gross    = 50_000.0   # mean annual earnings
    sigma_gross = 10_000.0   # standard deviation of earnings
    poverty_line = 20_000.0  # USD threshold

    # ---- Scenario A: Normal earnings -------------------------
    # Analytical answer:
    #   Disposable = 0.8 * Gross - 5000
    #   E[Disp]    = 0.8 * 50000 - 5000 = 35000
    #   SD[Disp]   = 0.8 * 10000 = 8000
    #   P(Disp < 20000) = P(Z < (20000 - 35000) / 8000)
    #                   = P(Z < -1.875)
    #                   ≈ 3.04%  (from standard normal table)
    #
    # We can verify this without any table using the formula above.

    incomes_normal = [random.gauss(mu_gross, sigma_gross) for _ in range(n)]
    disposable_normal = [worker_disposable_income(g) for g in incomes_normal]
    prob_poor_normal  = sum(1 for d in disposable_normal if d < poverty_line) / n

    # Compute moments of simulated disposable income
    d_mean = sample_mean(disposable_normal)
    d_std  = math.sqrt(sample_variance(disposable_normal, ddof=1))

    print(f"\n  Scenario A — Normal earnings N({mu_gross:,.0f}, {sigma_gross:,.0f}²)")
    print(f"  n = {n:,} simulated workers")
    print()
    print(f"  Simulated E[disposable income]  = ${d_mean:>10,.2f}")
    print(f"  Simulated SD[disposable income] = ${d_std:>10,.2f}")
    print()
    print(f"  P(disposable < poverty line):")
    print(f"    Simulated  : {prob_poor_normal * 100:.3f}%")
    print(f"    Analytical : ~3.04%  (from standard normal table)")

    # ---- Scenario B: Earnings drawn from a skewed distribution --
    # Replace Normal with a log-normal: if log(G) ~ N(mu_log, sigma_log),
    # then G has a right skew (mirrors real wage distributions).
    # We choose parameters so E[G] ≈ 50,000.
    #
    # For log-normal: E[G] = exp(mu_log + sigma_log²/2)
    # We want E[G] = 50000, Std[G] ≈ 10000.
    # Solve: mu_log + sigma_log²/2 = ln(50000)
    #        sigma_log² = ln(1 + (10000/50000)²) ≈ 0.0392
    sigma_log = math.sqrt(math.log(1 + (sigma_gross / mu_gross) ** 2))
    mu_log    = math.log(mu_gross) - sigma_log ** 2 / 2

    random.seed(seed)
    incomes_lognormal = [
        math.exp(random.gauss(mu_log, sigma_log)) for _ in range(n)
    ]
    disposable_lognormal = [worker_disposable_income(g) for g in incomes_lognormal]
    prob_poor_lognormal  = sum(
        1 for d in disposable_lognormal if d < poverty_line
    ) / n

    d_mean_ln = sample_mean(disposable_lognormal)
    d_std_ln  = math.sqrt(sample_variance(disposable_lognormal, ddof=1))

    print()
    print(f"  Scenario B — Log-normal earnings (same E[G] and SD[G])")
    print(f"  This distribution has a right tail — more realistic.")
    print()
    print(f"  Simulated E[disposable income]  = ${d_mean_ln:>10,.2f}")
    print(f"  Simulated SD[disposable income] = ${d_std_ln:>10,.2f}")
    print(f"  P(disposable < poverty line)    = {prob_poor_lognormal * 100:.3f}%")
    print()
    print("  Key takeaway: with a fat left tail (log-normal has less left mass),")
    print("  poverty risk actually decreases compared to the Normal scenario.")
    print("  Simulation made this comparison effortless — no tables needed.")


# ==============================================================
# main — run all sections in order
# ==============================================================

def main() -> None:
    section("1) Random Variables and Probability Distributions")
    demo_distributions(n=10, seed=42)

    section("2) Expected Value and Variance — Formula vs. Simulation")
    demo_ev_variance(seed=42)

    section("3) Law of Large Numbers")
    demo_lln(seed=42)

    section("4) Central Limit Theorem")
    demo_clt(n_samples=5_000, sample_size=30, seed=42)

    section("5) Monte Carlo Integration")
    demo_mc_integration()

    section("6) Monte Carlo Simulation — Worker Income / Poverty Risk")
    demo_mc_simulation(n=100_000, seed=42)

    print("\n" + "=" * 65)
    print("  Done. Run this script repeatedly — results are identical")
    print("  because every function sets its own random seed.")
    print("=" * 65)


if __name__ == "__main__":
    main()

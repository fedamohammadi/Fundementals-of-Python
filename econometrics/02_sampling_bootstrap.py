"""
Sampling and the Bootstrap
==========================
Concepts covered:
- Random sampling (with and without replacement)
- The bootstrap: resampling to estimate uncertainty
- Bootstrap standard errors and confidence intervals
- Comparing bootstrap SE to the analytical formula
"""

import random
import math


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ==============================================================
# 1. Random Sampling
# ==============================================================
# Sampling without replacement: each unit can only appear once.
# Sampling with replacement: the same unit can be picked again.
# The distinction matters enormously for inference.

def sample_mean(data: list[float]) -> float:
    return sum(data) / len(data)


def sample_std(data: list[float]) -> float:
    m = sample_mean(data)
    var = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(var)


def demo_sampling(seed: int = 42) -> None:
    random.seed(seed)

    population = list(range(1, 101))   # incomes 1..100 (thousands)
    n = 20

    # Without replacement: no duplicates, like a survey
    srs = random.sample(population, n)

    # With replacement: duplicates allowed, like the bootstrap
    wr  = [random.choice(population) for _ in range(n)]

    print(f"\n  Population: integers 1-100  (true mean = 50.5)")
    print(f"  Sample size: {n}")
    print()
    print(f"  Without replacement  mean={sample_mean(srs):.2f}  "
          f"std={sample_std(srs):.2f}")
    print(f"  With replacement     mean={sample_mean(wr):.2f}  "
          f"std={sample_std(wr):.2f}")
    print()
    print(f"  Duplicates in WR sample: "
          f"{n - len(set(wr))} out of {n} draws")


# ==============================================================
# 2. The Bootstrap
# ==============================================================
# Idea: treat your sample as if it were the population, then
# resample from it (with replacement) B times.  Each resample
# gives one estimate of your statistic.  The spread of those
# B estimates approximates the sampling distribution — without
# needing a formula.

def bootstrap(data: list[float], statistic, B: int = 2000, seed: int = 42):
    """Draw B bootstrap resamples and return the statistic for each."""
    random.seed(seed)
    n = len(data)
    return [
        statistic([random.choice(data) for _ in range(n)])
        for _ in range(B)
    ]


def demo_bootstrap(seed: int = 42) -> None:
    random.seed(seed)

    # Simulate a skewed income sample (log-normal, n=40)
    mu_log, sigma_log = 10.5, 0.6
    sample = [math.exp(random.gauss(mu_log, sigma_log)) for _ in range(40)]

    true_pop_mean = math.exp(mu_log + sigma_log ** 2 / 2)

    boot_means = bootstrap(sample, sample_mean, B=3000, seed=1)

    boot_se   = sample_std(boot_means)
    boot_mean = sample_mean(boot_means)

    print(f"\n  Sample size : 40   (log-normal, skewed)")
    print(f"  True E[X]   : {true_pop_mean:,.0f}")
    print(f"  Sample mean : {sample_mean(sample):,.0f}")
    print()
    print(f"  Bootstrap (B=3,000 resamples):")
    print(f"    Mean of boot. means : {boot_mean:,.0f}")
    print(f"    Bootstrap SE        : {boot_se:,.0f}")


# ==============================================================
# 3. Bootstrap Confidence Intervals
# ==============================================================
# Two common approaches:
#
# Percentile CI: just take the alpha/2 and 1-alpha/2 quantiles
#   of the bootstrap distribution.
#   CI = [Q(alpha/2), Q(1-alpha/2)]
#
# Normal CI (for comparison): uses the analytical SE formula
#   CI = [x_bar - 1.96*SE, x_bar + 1.96*SE]
#   where SE = std / sqrt(n)
#
# The percentile bootstrap requires no distributional assumption,
# which is why it shines for skewed data or unusual statistics.

def percentile_ci(boot_stats: list[float], alpha: float = 0.05):
    """Return the (alpha/2, 1-alpha/2) percentile bootstrap CI."""
    sorted_stats = sorted(boot_stats)
    n = len(sorted_stats)
    lo_idx = int(math.floor(alpha / 2 * n))
    hi_idx = int(math.ceil((1 - alpha / 2) * n)) - 1
    return sorted_stats[lo_idx], sorted_stats[hi_idx]


def demo_bootstrap_ci(seed: int = 42) -> None:
    random.seed(seed)

    # Same log-normal sample as Section 2
    mu_log, sigma_log = 10.5, 0.6
    sample = [math.exp(random.gauss(mu_log, sigma_log)) for _ in range(40)]

    true_mean = math.exp(mu_log + sigma_log ** 2 / 2)
    x_bar     = sample_mean(sample)
    se_analytical = sample_std(sample) / math.sqrt(len(sample))

    # Bootstrap percentile CI
    boot_means = bootstrap(sample, sample_mean, B=5000, seed=7)
    ci_boot_lo, ci_boot_hi = percentile_ci(boot_means, alpha=0.05)

    # Normal-theory CI (assumes sampling dist. is normal)
    ci_norm_lo = x_bar - 1.96 * se_analytical
    ci_norm_hi = x_bar + 1.96 * se_analytical

    print(f"\n  True mean         : {true_mean:>10,.0f}")
    print(f"  Sample mean       : {x_bar:>10,.0f}")
    print()
    print(f"  95% Percentile bootstrap CI : "
          f"[{ci_boot_lo:>8,.0f}, {ci_boot_hi:>8,.0f}]")
    print(f"  95% Normal-theory CI        : "
          f"[{ci_norm_lo:>8,.0f}, {ci_norm_hi:>8,.0f}]")
    print()
    print(f"  True mean inside bootstrap CI : "
          f"{'YES' if ci_boot_lo <= true_mean <= ci_boot_hi else 'NO'}")
    print(f"  True mean inside normal CI    : "
          f"{'YES' if ci_norm_lo <= true_mean <= ci_norm_hi else 'NO'}")

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

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

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

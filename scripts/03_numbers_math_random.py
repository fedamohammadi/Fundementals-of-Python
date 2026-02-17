"""
03_numbers_math_random.py

Purpose:
--------
Learn how Python handles numbers and basic math, and how randomness works.
These ideas show up everywhere in research: simulation, bootstrapping,
Monte Carlo, and basic data transformations.

Concepts covered (progressively):
- int vs float
- Arithmetic operators
- Division pitfalls
- Rounding
- math module basics
- Randomness with random and numpy
- Reproducibility with seeds

"""

# ==============================================================
# Add numbers/math/random lesson skeleton
# ==============================================================

import math
import random


def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Numbers, Math, and Randomness")
    print("This lesson will cover numeric types, math operations, and randomness.")


# ==============================================================
# Add numeric types and arithmetic operations
# ==============================================================
def main() -> None:
    section("1) Numeric types")

    a = 10          # int
    b = 3.5         # float

    print(f"a = {a} (type: {type(a)})")
    print(f"b = {b} (type: {type(b)})")

    section("2) Arithmetic operators")

    x = 7
    y = 3

    print(f"x + y = {x + y}")
    print(f"x - y = {x - y}")
    print(f"x * y = {x * y}")
    print(f"x / y = {x / y}")      # true division (float)
    print(f"x // y = {x // y}")    # floor division (int)
    print(f"x % y = {x % y}")      # remainder
    print(f"x ** y = {x ** y}")    # exponent




if __name__ == "__main__":
    main()

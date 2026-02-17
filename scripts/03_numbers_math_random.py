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
# Added numbers/math/random lesson skeleton
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
# Added numeric types and arithmetic operations
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

# ==============================================================
# Added rounding and math module basics
# ==============================================================
def main() -> None:
    section("1) Numeric types")

    a = 10
    b = 3.5

    print(f"a = {a} (type: {type(a)})")
    print(f"b = {b} (type: {type(b)})")

    section("2) Arithmetic operators")

    x = 7
    y = 3

    print(f"x + y = {x + y}")
    print(f"x - y = {x - y}")
    print(f"x * y = {x * y}")
    print(f"x / y = {x / y}")
    print(f"x // y = {x // y}")
    print(f"x % y = {x % y}")
    print(f"x ** y = {x ** y}")

    section("3) Rounding and formatting")

    value = 1234.56789
    print(f"value = {value}")
    print(f"round(value, 2) = {round(value, 2)}")
    print(f"Formatted to 2 decimals: {value:.2f}")
    print(f"Formatted with commas: {value:,.2f}")

    section("4) math module basics")

    z = 16
    print(f"sqrt({z}) = {math.sqrt(z)}")
    print(f"log({z}) (natural log) = {math.log(z)}")
    print(f"log10({z}) = {math.log10(z)}")
    print(f"exp(1) = {math.exp(1)}")
    print(f"pi = {math.pi}")


# ==============================================================
# Added randomness + reproducibility with seeds
# ==============================================================
def main() -> None:
    section("1) Numeric types")

    a = 10
    b = 3.5

    print(f"a = {a} (type: {type(a)})")
    print(f"b = {b} (type: {type(b)})")

    section("2) Arithmetic operators")

    x = 7
    y = 3

    print(f"x + y = {x + y}")
    print(f"x - y = {x - y}")
    print(f"x * y = {x * y}")
    print(f"x / y = {x / y}")
    print(f"x // y = {x // y}")
    print(f"x % y = {x % y}")
    print(f"x ** y = {x ** y}")

    section("3) Rounding and formatting")

    value = 1234.56789
    print(f"value = {value}")
    print(f"round(value, 2) = {round(value, 2)}")
    print(f"Formatted to 2 decimals: {value:.2f}")
    print(f"Formatted with commas: {value:,.2f}")

    section("4) math module basics")

    z = 16
    print(f"sqrt({z}) = {math.sqrt(z)}")
    print(f"log({z}) (natural log) = {math.log(z)}")
    print(f"log10({z}) = {math.log10(z)}")
    print(f"exp(1) = {math.exp(1)}")
    print(f"pi = {math.pi}")

    section("5) Randomness (why seeds matter)")

    # Without a seed, results change every run.
    print("Random integers (no seed):")
    print([random.randint(1, 10) for _ in range(5)])

    # Setting a seed makes randomness reproducible.
    random.seed(42)
    print("\nRandom integers (seed=42):")
    print([random.randint(1, 10) for _ in range(5)])

    section("6) Mini simulation idea (coin flips)")

    random.seed(7)
    flips = 20
    heads = 0

    for _ in range(flips):
        if random.random() < 0.5:
            heads += 1

    tails = flips - heads
    print(f"Flips: {flips}, Heads: {heads}, Tails: {tails}")
    print(f"Head share: {heads / flips:.2f}")

    section("Done")
    print("Next lesson: booleans and conditionals (decision logic).")


if __name__ == "__main__":
    main()

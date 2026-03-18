"""
Concepts covered (progressively):
- What a module is
- What a package is
- Basic import styles
- Import aliases
- Importing specific functions
- Why import style matters
"""

import math
import random


def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Modules, Packages, and Imports")
    print(" Here you see how Python organizes and imports code.")

# ==============================================================
# Basic import examples
# ==============================================================

def main() -> None:
    section("1) Importing full modules")

    print(f"math.sqrt(16) = {math.sqrt(16)}")
    print(f"math.pi = {math.pi}")

    random.seed(42)
    print(f"random.randint(1, 10) = {random.randint(1, 10)}")

    section("2) Why modules matter")

    print("Modules help organize related functions and tools.")
    print("For example, the math module contains many mathematical functions.")



if __name__ == "__main__":
    main()
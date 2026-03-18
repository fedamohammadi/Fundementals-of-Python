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

    section("2) Why modules are good")

    print("Modules help organize related functions and tools.")
    print("For example, the math module contains many mathematical functions.")

# ==============================================================
# import styles and aliases
# ==============================================================

from math import sqrt, log
import statistics as stats


def main() -> None:
    section("1) Different import styles")

    # Full module import
    print(f"math.sqrt(25) = {math.sqrt(25)}")

    # Import specific functions directly
    print(f"sqrt(36) = {sqrt(36)}")
    print(f"log(10) = {log(10):.4f}")

    # Alias import keeps code shorter
    data = [10, 20, 30, 40]
    print(f"Mean using statistics alias: {stats.mean(data)}")

    section("2) Why aliases are useful")

    print("Aliases are common when module names are long.")
    print("For example, pandas is usually imported as pd and numpy as np.")


# ==============================================================
# Modules vs packages explanation
# ==============================================================

from pathlib import Path


def main() -> None:
    section("1) What is a module?")

    print("A module is a single Python file that contains code.")
    print("Examples: math, random, pathlib")

    section("2) What is a package?")

    print("A package is a folder containing multiple Python modules.")
    print("Large libraries like pandas and statsmodels are packages.")

    section("3) Different import styles")

    print(f"math.sqrt(25) = {math.sqrt(25)}")
    print(f"sqrt(36) = {sqrt(36)}")
    print(f"log(10) = {log(10):.4f}")

    data = [10, 20, 30, 40]
    print(f"Mean using statistics alias: {stats.mean(data)}")

    section("4) Another useful standard library import")

    # pathlib helps you work with file paths in a safer and cleaner way
    data_path = Path("data") / "wage1.csv"
    print(f"Path object: {data_path}")


if __name__ == "__main__":
    main()




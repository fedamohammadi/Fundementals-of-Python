# ==============================================================
# Functions/scope lesson skeleton
# ==============================================================

"""
Purpose:
--------
Learn how to define and use functions properly.
Understand local vs global scope.

Functions are important in research code because they:
- Avoid repetition
- Make logic reusable
- Improve clarity
- Help structure complex workflows

Concepts covered:
- Defining functions
- Parameters and return values
- Default arguments
- Local vs global scope
- Why scope matters in research code
"""

def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ===============================
# Main
# ===============================

def main() -> None:
    section("Functions and Scope")
    print("This doc introduces reusable functions and variable scope.")


# ==============================================================
# Basic function definitions and returns
# ==============================================================

def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def compute_average(values: list[float]) -> float:
    """Return the average of a list of numbers."""
    return sum(values) / len(values)


def main() -> None:
    section("1) Basic function definitions")

    result = add(5, 3)
    print(f"add(5, 3) = {result}")

    incomes = [32000, 41000, 29000, 50000]
    avg_income = compute_average(incomes)
    print(f"Average income: {avg_income:.2f}")





if __name__ == "__main__":
    main()
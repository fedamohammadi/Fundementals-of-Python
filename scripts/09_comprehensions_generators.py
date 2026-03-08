"""
Purpose:
--------
Learn comprehensions and generators.

These are extremely common in data analysis because they allow
you to transform, filter, and build collections in a clean way.

Concepts covered:
- List comprehensions
- Conditional comprehensions
- Dictionary comprehensions
- Set comprehensions
- Generator expressions
- Why generators are memory efficient
"""

def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Comprehensions and Generators")
    print("This one introduces powerful Python patterns for transforming data.")


# ==============================================================
# List comprehensions
# ==============================================================

def main() -> None:
    section("1) Basic list comprehension")

    numbers = [1, 2, 3, 4, 5]

    # Square every number
    squares = [x**2 for x in numbers]

    print(f"numbers: {numbers}")
    print(f"squares: {squares}")

    section("2) Filtering with list comprehensions")

    values = [10, -5, 20, -2, 30]

    # Keep only positive values
    positives = [x for x in values if x > 0]

    print(f"values: {values}")
    print(f"positives: {positives}")

    section("3) Transform and filter together")

    # Convert incomes to thousands and keep those above 40k
    incomes = [32000, 41000, 29000, 50000]
    high_income_k = [x / 1000 for x in incomes if x > 40000]

    print(f"incomes: {incomes}")
    print(f"income (thousands) above 40k: {high_income_k}")
    
    

if __name__ == "__main__":
    main()
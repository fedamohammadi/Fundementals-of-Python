"""
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
    
# ==============================================================
# Dictionary and set comprehensions
# ==============================================================

def main() -> None:
    section("1) List comprehension")

    numbers = [1, 2, 3, 4, 5]
    squares = [x**2 for x in numbers]

    print(f"squares: {squares}")

    section("2) Dictionary comprehension")

    states = ["KY", "VA", "OH"]

    # Map each state to an index value
    state_index = {state: i for i, state in enumerate(states)}

    print(f"states: {states}")
    print(f"state_index: {state_index}")

    section("3) Set comprehension")

    values = [1, 2, 2, 3, 3, 4]

    # Extract unique squared values
    unique_squares = {x**2 for x in values}

    print(f"values: {values}")
    print(f"unique_squares: {unique_squares}")


# ==============================================================
# Generator expressions
# ==============================================================

def main() -> None:
    section("1) List comprehension")

    numbers = [1, 2, 3, 4, 5]
    squares = [x**2 for x in numbers]

    print(f"squares: {squares}")

    section("2) Dictionary comprehension")

    states = ["KY", "VA", "OH"]
    state_index = {state: i for i, state in enumerate(states)}

    print(f"state_index: {state_index}")

    section("3) Set comprehension")

    values = [1, 2, 2, 3, 3, 4]
    unique_squares = {x**2 for x in values}

    print(f"unique_squares: {unique_squares}")

    section("4) Generator expressions")

    numbers = range(1, 6)

    # Generator expressions use parentheses instead of brackets
    # They generate values lazily (one at a time).
    squares_gen = (x**2 for x in numbers)

    print("Generator output:")
    for val in squares_gen:
        print(val)

    # Example: sum of squares without storing everything in memory
    big_sum = sum(x**2 for x in range(1, 1_000_001))

    print("Computed sum of squares for 1..1,000,000 using a generator.")
    print(f"Result: {big_sum}")


if __name__ == "__main__":
    main()
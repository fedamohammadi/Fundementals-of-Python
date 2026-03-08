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
    print("This lesson introduces powerful Python patterns for transforming data.")




if __name__ == "__main__":
    main()
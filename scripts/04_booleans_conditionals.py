"""
04_booleans_conditionals.py

Purpose:
--------
Learn boolean logic and conditional statements.
This is essential for research work because you constantly make decisions like:
- filter rows
- handle missing values
- apply rules based on thresholds
- build indicators (0/1 variables)

Concepts covered (progressively):
- bool type and comparisons
- Logical operators: and, or, not
- if / elif / else
- Common patterns: threshold rules, missing checks, category mapping
"""

# ===============================
# Helpers
# ===============================

def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ===============================
# Main
# ===============================

def main() -> None:
    section("Booleans and Conditionals")
    print("This lesson will introduce boolean logic and if-statements.")


# ==============================================================
# Added comparisons and logical operators
# ==============================================================
def main() -> None:
    section("1) Boolean type and comparisons")

    x = 10
    y = 7

    print(f"x = {x}, y = {y}")
    print(f"x > y  -> {x > y}")
    print(f"x >= y -> {x >= y}")
    print(f"x < y  -> {x < y}")
    print(f"x == y -> {x == y}")
    print(f"x != y -> {x != y}")

    section("2) Logical operators: and / or / not")

    a = True
    b = False

    print(f"a = {a}, b = {b}")
    print(f"a and b -> {a and b}")
    print(f"a or b  -> {a or b}")
    print(f"not a   -> {not a}")

    section("3) Common pattern: range check")

    age = 19
    is_college_age = (age >= 18) and (age <= 24)
    print(f"age = {age}")
    print(f"is_college_age (18-24)? {is_college_age}")







if __name__ == "__main__":
    main()

# ==============================================================
# Errors/exceptions lesson skeleton
# ==============================================================

"""
Purpose:
--------
Learn how to handle errors using exceptions.
In research code, errors happen all the time:
- messy data
- missing values
- unexpected types
- divide-by-zero problems
- file paths that do not exist

Concepts covered:
- Common error types
- try / except
- Multiple except blocks
- else and finally
- Raising errors intentionally (fail fast)
"""

def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Errors and Exceptions")
    print("Here you should see try/except and safe error handling.")


# ==============================================================
# Basic try/except examples
# ==============================================================

def main() -> None:
    section("1) A common error: dividing by zero")

    x = 10
    y = 0

    try:
        result = x / y
        print(f"result = {result}")
    except ZeroDivisionError:
        print("You cannot divide by zero. Handle this case in your code.")

    section("2) Another common error: invalid type conversion")

    text = "not_a_number"

    try:
        value = int(text)
        print(f"value = {value}")
    except ValueError:
        print(f"Cannot convert '{text}' to an integer.")






if __name__ == "__main__":
    main()

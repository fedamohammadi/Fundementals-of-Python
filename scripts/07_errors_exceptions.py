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


# ==============================================================
# Multiple except blocks and safe parsing function
# ==============================================================

def safe_int(text: str, default: int | None = None) -> int | None:
    """
    Convert text to int safely.
    If conversion fails, return default.
    """
    try:
        return int(text)
    except ValueError:
        return default


def main() -> None:
    section("1) Multiple exception types")

    values = ["10", "5", "bad", None]

    for v in values:
        try:
            # This can raise TypeError if v is None
            # or ValueError if v is not numeric text.
            parsed = int(v)  # type: ignore[arg-type]
            print(f"Parsed {v} -> {parsed}")
        except ValueError:
            print(f"ValueError: cannot parse '{v}'")
        except TypeError:
            print("TypeError: got None instead of text")

    section("2) Safer approach: use a helper function")

    for v in ["10", "bad", "30"]:
        print(f"safe_int('{v}') -> {safe_int(v, default=-1)}")




if __name__ == "__main__":
    main()

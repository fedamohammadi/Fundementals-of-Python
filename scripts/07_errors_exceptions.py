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
    print("This lesson will introduce try/except and safe error handling.")




if __name__ == "__main__":
    main()

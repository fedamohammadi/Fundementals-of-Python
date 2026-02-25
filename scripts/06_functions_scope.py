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








if __name__ == "__main__":
    main()
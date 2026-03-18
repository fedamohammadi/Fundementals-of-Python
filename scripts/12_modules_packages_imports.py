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
    print("This lesson introduces how Python organizes and imports code.")




if __name__ == "__main__":
    main()
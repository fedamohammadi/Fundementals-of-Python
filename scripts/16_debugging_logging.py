"""
- print debugging
- using assertions
- basic logging with logging module
- different logging levels
"""

import logging


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Debugging and Logging")
    print("This lesson introduces basic debugging techniques.")



if __name__ == "__main__":
    main()



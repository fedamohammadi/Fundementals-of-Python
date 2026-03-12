"""
Concepts covered:
- Creating dates and datetimes
- Formatting dates as strings
- Parsing strings into dates
- Timedelta arithmetic
- Iterating over time periods
- Basic time series style examples
"""

from datetime import date, datetime, timedelta


def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Datetime and Time Series Basics")


# ==============================================================
# Creating dates and formatting
# ==============================================================

def main() -> None:
    section("1) Creating dates and datetimes")

    today = date(2026, 3, 11)
    now = datetime(2026, 3, 11, 14, 30, 0)

    print(f"today = {today}")
    print(f"now = {now}")

    section("2) Accessing parts of a date")

    print(f"Year: {today.year}")
    print(f"Month: {today.month}")
    print(f"Day: {today.day}")

    section("3) Formatting dates as strings")

    print(today.strftime("%Y-%m-%d"))
    print(now.strftime("%B %d, %Y"))
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
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
    print("This lesson introduces Python dates, times, and simple time-based workflows.")





if __name__ == "__main__":
    main()
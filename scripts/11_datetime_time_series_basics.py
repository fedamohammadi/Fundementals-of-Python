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

# ==============================================================
# Parsing and timedelta arithmetic
# ==============================================================

    today = date(2026, 3, 11)
    now = datetime(2026, 3, 11, 14, 30, 0)

    print(f"today = {today}")
    print(f"now = {now}")

    section("2) Parsing strings into dates")

    raw_date = "2026-03-11"
    parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()

    print(f"raw_date = {raw_date}")
    print(f"parsed_date = {parsed_date}")

    section("3) Timedelta arithmetic")

    one_week = timedelta(days=7)
    next_week = today + one_week
    last_week = today - one_week

    print(f"today = {today}")
    print(f"next_week = {next_week}")
    print(f"last_week = {last_week}")

    section("4) Days between dates")

    start = date(2026, 1, 1)
    end = date(2026, 3, 11)

    diff = end - start
    print(f"Days between {start} and {end}: {diff.days}")




if __name__ == "__main__":
    main()
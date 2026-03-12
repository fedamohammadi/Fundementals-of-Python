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


# ==============================================================
# A simple time series style examples
# ==============================================================

    one_week = timedelta(days=7)
    print(f"today = {today}")
    print(f"today + 7 days = {today + one_week}")
    print(f"today - 7 days = {today - one_week}")

    section("4) Building a simple time series")

    # A simple list of monthly observations
    dates = [
        date(2025, 10, 1),
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]
    inflation_rates = [0.021, 0.023, 0.024, 0.026, 0.025]

    for d, rate in zip(dates, inflation_rates):
        print(f"{d}: {rate:.2%}")

    section("5) Example: month-to-month change")

    # Computing simple changes between consecutive values
    changes = []
    for i in range(1, len(inflation_rates)):
        change = inflation_rates[i] - inflation_rates[i - 1]
        changes.append(change)

    print("Month-to-month changes:")
    for i, change in enumerate(changes, start=1):
        print(f"{dates[i - 1]} -> {dates[i]}: {change:+.2%}")





if __name__ == "__main__":
    main()
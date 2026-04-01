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

# ==============================================================
# Print debugging and assertions
# ==============================================================

def main() -> None:
    section("1) Print debugging")

    values = [10, 20, 30]

    total = 0
    for v in values:
        print(f"Adding {v} to total={total}")
        total += v

    print(f"Final total: {total}")

    section("2) Assertions")

    result = total / len(values)

    # Assert checks conditions during development
    assert result > 0, "Average should be positive"

    print(f"Average: {result}")

# ==============================================================
# logging basics
# ==============================================================

def main() -> None:
    section("1) Logging setup")

    logging.basicConfig(level=logging.INFO)

    logging.info("Program started")
    logging.warning("This is a warning example")

    section("2) Logging inside logic")

    values = [10, 0, 30]

    for v in values:
        if v == 0:
            logging.error("Encountered zero value")
        else:
            logging.info(f"Processing value {v}")


# ==============================================================
# Structured logging example
# ==============================================================

def compute_average(values: list[float]) -> float:
    if len(values) == 0:
        logging.error("Empty list provided to compute_average")
        raise ValueError("Cannot compute average of empty list")

    return sum(values) / len(values)


def main() -> None:
    section("1) Logging setup")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(message)s"
    )

    logging.info("Program started")

    section("2) Safe computation with logging")

    data = [10.0, 20.0, 30.0]

    try:
        avg = compute_average(data)
        logging.info(f"Computed average: {avg:.2f}")
    except ValueError as e:
        logging.error(f"Error: {e}")

    section("3) Example with bad input")

    bad_data: list[float] = []

    try:
        avg = compute_average(bad_data)
    except ValueError as e:
        logging.error(f"Handled error: {e}")



if __name__ == "__main__":
    main()



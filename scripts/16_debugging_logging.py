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




if __name__ == "__main__":
    main()



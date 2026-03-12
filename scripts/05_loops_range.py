# ==============================================================
# loops/range
# ==============================================================
"""
Concepts covered:
- for loops
- while loops
- range()
- break and continue
- loop patterns used in research (accumulators, counters)
"""


def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==============================================================
# for-loops and range basics
# ==============================================================
def main() -> None:
    section("1) for-loops with a list")

    states = ["Kentucky", "Virginia", "Ohio"]
    for state in states:
        print(f"State: {state}")

    section("2) range() basics")

    print("range(5):")
    for i in range(5):
        print(i)

    print("\nrange(2, 7):")
    for i in range(2, 7):
        print(i)

    print("\nrange(0, 10, 2) (step size = 2):")
    for i in range(0, 10, 2):
        print(i)

# ==============================================================
# Accumulators, break/continue, and while-loops
# ==============================================================
def main() -> None:
    section("1) for-loops with a list")

    states = ["Kentucky", "Virginia", "Ohio"]
    for state in states:
        print(f"State: {state}")

    section("2) range() basics")

    for i in range(5):
        print(f"i = {i}")

    section("3) Accumulator pattern (common in research)")

    incomes = [32000, 41000, 29000, 50000]
    total_income = 0

    for income in incomes:
        total_income += income

    avg_income = total_income / len(incomes)
    print(f"Incomes: {incomes}")
    print(f"Total income: {total_income}")
    print(f"Average income: {avg_income:.2f}")

    section("4) break and continue")

    nums = [3, 7, -2, 10, 5]

    print("Stop when we see a negative number (break):")
    for n in nums:
        if n < 0:
            print(f"Found negative {n}, stopping.")
            break
        print(f"n = {n}")

    print("\nSkip small numbers below 5 (continue):")
    for n in nums:
        if n < 5:
            continue
        print(f"n = {n}")

    section("5) while-loop basics")

    count = 0
    while count < 3:
        print(f"count = {count}")
        count += 1

# ==============================================================
# A simple simulation loop
# ==============================================================
def main() -> None:
    section("1) for-loops with a list")

    states = ["Kentucky", "Virginia", "Ohio"]
    for state in states:
        print(f"State: {state}")

    section("2) range() basics")

    for i in range(5):
        print(f"i = {i}")

    section("3) Accumulator pattern (common in research)")

    incomes = [32000, 41000, 29000, 50000]
    total_income = 0

    for income in incomes:
        total_income += income

    avg_income = total_income / len(incomes)
    print(f"Incomes: {incomes}")
    print(f"Total income: {total_income}")
    print(f"Average income: {avg_income:.2f}")

    section("4) break and continue")

    nums = [3, 7, -2, 10, 5]

    print("Stop when we see a negative number (break):")
    for n in nums:
        if n < 0:
            print(f"Found negative {n}, stopping.")
            break
        print(f"n = {n}")

    print("\nSkip small numbers below 5 (continue):")
    for n in nums:
        if n < 5:
            continue
        print(f"n = {n}")

    section("5) while-loop basics")

    count = 0
    while count < 3:
        print(f"count = {count}")
        count += 1

    section("6) Mini simulation: estimate a probability")

    # Estimate the probability that a random integer from 1..100 is divisible by 7.
    import random
    random.seed(42)

    trials = 1000
    hits = 0

    for _ in range(trials):
        draw = random.randint(1, 100)
        if draw % 7 == 0:
            hits += 1

    prob_estimate = hits / trials
    print(f"Trials: {trials}")
    print(f"Hits (divisible by 7): {hits}")
    print(f"Estimated probability: {prob_estimate:.3f}")





if __name__ == "__main__":
    main()
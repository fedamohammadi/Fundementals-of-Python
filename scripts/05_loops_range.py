# ==============================================================
# loops/range lesson skeleton
# ==============================================================
"""
Purpose:
--------
Learn how to repeat actions using loops.
Loops are used constantly in research for:
- iterating over rows (when you really must)
- running simulations
- bootstrapping
- cleaning multiple variables
- applying rules to many items

Concepts covered (progressively):
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


# ===============================
# Main
# ===============================

def main() -> None:
    section("Loops and range()")
    print("This lesson will introduce for-loops, while-loops, and range().")

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







if __name__ == "__main__":
    main()
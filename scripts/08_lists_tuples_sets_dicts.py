# ==============================================================
# Data structures
# ==============================================================

"""
Purpose:
--------
Learn the four core Python data structures:
- Lists
- Tuples
- Sets
- Dictionaries

These structures are essential for:
- Storing datasets
- Cleaning messy values
- Managing categorical variables
- Representing observations and features
"""


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Data Structures in Python")
    print("This lesson covers lists, tuples, sets, and dictionaries.")


# ==============================================================
# Lists (core data container)
# ==============================================================

def main() -> None:
    section("1) Lists")

    incomes = [32000, 41000, 29000, 50000]

    print(f"incomes = {incomes}")
    print(f"First income: {incomes[0]}")
    print(f"Last income: {incomes[-1]}")

    section("List modification")

    incomes.append(45000)  # Add new value
    print(f"After append: {incomes}")

    incomes.remove(29000)  # Remove specific value
    print(f"After remove: {incomes}")

    print(f"Sorted incomes: {sorted(incomes)}")

    section("List comprehension (very important)")

    # Multiply each income by 1.05 (5% growth)
    grown = [x * 1.05 for x in incomes]
    print(f"Grown incomes: {grown}")




if __name__ == "__main__":
    main()
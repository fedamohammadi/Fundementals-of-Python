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


# ==============================================================
# tuples and sets
# ==============================================================

def main() -> None:
    section("1) Lists")

    incomes = [32000, 41000, 29000, 50000]
    print(f"incomes = {incomes}")

    section("2) Tuples")

    # Tuples are immutable (cannot change after creation)
    coordinates = (37.57, -84.29)
    print(f"coordinates = {coordinates}")
    print(f"Latitude: {coordinates[0]}")

    # coordinates[0] = 0  # This would raise an error

    section("3) Sets")

    states = ["KY", "CA", "KY", "TX", "CA"]
    unique_states = set(states)  # Removes duplicates
    print(f"Original states: {states}")
    print(f"Unique states: {unique_states}")

    section("Set operations")

    a = {"KY", "CA", "TX"}
    b = {"TX", "NY"}

    print(f"Union: {a | b}")
    print(f"Intersection: {a & b}")

# ==============================================================
# Dictionaries (most important structure)
# ==============================================================

def main() -> None:
    section("1) Lists")

    incomes = [32000, 41000, 29000, 50000]
    print(f"incomes = {incomes}")

    section("2) Tuples")

    coordinates = (37.57, -84.29)
    print(f"coordinates = {coordinates}")

    section("3) Sets")

    states = ["KY", "CA", "KY", "TX", "CA"]
    print(f"Unique states: {set(states)}")

    section("4) Dictionaries (key-value pairs)")

    # Dictionary represents structured data
    observation = {
        "state": "Kentucky",
        "year": 2023,
        "income": 42000,
        "unemployment_rate": 0.041
    }

    print(f"Observation: {observation}")
    print(f"Income: {observation['income']}")

    section("Modify dictionary")

    observation["income"] = 43000  # Update value
    observation["population"] = 4500000  # Add new key

    print(f"Updated observation: {observation}")

    section("Loop through dictionary")

    for key, value in observation.items():
        print(f"{key} -> {value}")

    section("Why dictionaries are important")

    print("In data analysis, each row of data is often represented as a dictionary.")
    print("Later, pandas will generalize this idea into DataFrames.")



if __name__ == "__main__":
    main()








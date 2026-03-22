"""
- Basic type hints
- Function annotations
- Return types
- List and dict type hints
- Naming and formatting style
- Why readability matters

"""


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)



def main() -> None:
    print("Type Hints and Style")


# ==============================================================
# Basic function type hints
# ==============================================================

def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    return f"Hello, {name}!"


def main() -> None:
    section("1) Basic type hints")

    print(add(3, 5))
    print(greet("Feda"))

    section("2) Why type hints help")

    print("Type hints make code easier to understand.")
    print("They also help editors catch mistakes earlier.")


# ==============================================================
# Collection type hints and cleaner naming
# ==============================================================

def compute_average(values: list[float]) -> float:
    return sum(values) / len(values)


def count_states(states: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for state in states:
        if state in counts:
            counts[state] += 1
        else:
            counts[state] = 1

    return counts


def main() -> None:
    section("1) Type hints with collections")

    incomes = [32000.0, 41000.0, 29000.0, 50000.0]
    print(f"Average income: {compute_average(incomes):.2f}")

    states = ["KY", "VA", "KY", "OH", "VA", "KY"]
    print(f"State counts: {count_states(states)}")

    section("2) Style example")

    # Good names make code readable without needing many comments.
    student_name = "Feda"
    major_field = "Economics"

    print(f"Student: {student_name}")
    print(f"Major: {major_field}")




if __name__ == "__main__":
    main()
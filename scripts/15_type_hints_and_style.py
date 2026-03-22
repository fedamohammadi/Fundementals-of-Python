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


if __name__ == "__main__":
    main()
"""
01_variables_types.py

Purpose:
--------
Learn what variables are, what "types" are, and why types matter in data work.

Concepts covered:
- Assigning variables
- Common Python types: int, float, str, bool
- Type inspection using type()
- Type conversion (casting)
- Core collections: list, tuple, dict, set
- Clean, readable printing

"""


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("Variables and Types")

    print("This lesson will introduce Python variables and basic data types.")


def main() -> None:
    section("1) Variables and Basic Types")

    age = 21                  # int
    gpa = 3.65                # float
    name = "Feda"             # str
    is_student = True         # bool

    print(f"age = {age} (type: {type(age)})")
    print(f"gpa = {gpa} (type: {type(gpa)})")
    print(f"name = {name} (type: {type(name)})")
    print(f"is_student = {is_student} (type: {type(is_student)})")




if __name__ == "__main__":
    main()


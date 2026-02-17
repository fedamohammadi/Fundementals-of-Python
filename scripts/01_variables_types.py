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
    section("1) Variables and basic types")

    age = 21                    # int
    gpa = 3.65                  # float
    name = "Feda"               # str
    is_student = True           # bool

    print(f"age = {age} (type: {type(age)})")
    print(f"gpa = {gpa} (type: {type(gpa)})")
    print(f"name = {name} (type: {type(name)})")
    print(f"is_student = {is_student} (type: {type(is_student)})")

    section("2) Why types matter (a common mistake)")

    x = "10"
    y = 5

    print(f"x = {x} (type: {type(x)})")
    print(f"y = {y} (type: {type(y)})")

    print("\nIf x is a string, x + '5' becomes string concatenation:")
    print(f"x + '5' = {x + '5'}")

    print("\nTo do math, convert x to an integer:")
    x_int = int(x)
    print(f"int(x) + y = {x_int + y}")

    section("3) Core collections (used constantly in research)")

    numbers_list = [1, 2, 3, 4]
    numbers_tuple = (1, 2, 3, 4)
    person_dict = {"name": "Feda", "age": 21, "gpa": 3.65}
    unique_set = {1, 2, 2, 3, 3, 3}

    print(f"numbers_list = {numbers_list} (type: {type(numbers_list)})")
    print(f"numbers_tuple = {numbers_tuple} (type: {type(numbers_tuple)})")
    print(f"person_dict = {person_dict} (type: {type(person_dict)})")
    print(f"unique_set = {unique_set} (type: {type(unique_set)})")

    section("4) Quick mini-exercises (no user input)")

    income_annual = 42000
    months = 12
    income_monthly = income_annual / months

    print(f"Annual income: {income_annual}")
    print(f"Monthly income: {income_monthly:.2f}")

    # Example: a simple boolean check
    threshold = 3500
    is_above_threshold = income_monthly > threshold
    print(f"Is monthly income above {threshold}? {is_above_threshold}")

    section("Done")
    print("Next: strings, formatting, and working cleanly with text data.")


if __name__ == "__main__":
    main()

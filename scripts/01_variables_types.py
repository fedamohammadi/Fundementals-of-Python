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

def main() -> None:
    section("1) Variables and Basic Types")

    age = 21
    gpa = 3.65
    name = "Feda"
    is_student = True

    print(f"age = {age} (type: {type(age)})")
    print(f"gpa = {gpa} (type: {type(gpa)})")
    print(f"name = {name} (type: {type(name)})")
    print(f"is_student = {is_student} (type: {type(is_student)})")

    section("2) Type Conversion")

    x = "10"
    y = 5

    print(f"x = {x} (type: {type(x)})")
    print(f"y = {y} (type: {type(y)})")

    x_int = int(x)
    print(f"int(x) + y = {x_int + y}")

    section("3) Core Collections")

    numbers_list = [1, 2, 3, 4]
    numbers_tuple = (1, 2, 3, 4)
    person_dict = {"name": "Feda", "age": 21}
    unique_set = {1, 2, 2, 3}

    print(f"List: {numbers_list} (type: {type(numbers_list)})")
    print(f"Tuple: {numbers_tuple} (type: {type(numbers_tuple)})")
    print(f"Dictionary: {person_dict} (type: {type(person_dict)})")
    print(f"Set (duplicates removed): {unique_set}")



if __name__ == "__main__":
    main()


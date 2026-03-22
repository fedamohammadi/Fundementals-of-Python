"""
- Defining a class
- Creating objects
- Instance attributes
- Instance methods
- __init__
- __repr__

"""

def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)



def main() -> None:
    section("OOP Minimum for Research")
 

# ==============================================================
# Basic class and object examples
# ==============================================================

class Student:
    def __init__(self, name: str, major: str) -> None:
        self.name = name
        self.major = major


def main() -> None:
    section("1) Create a simple class")

    student_1 = Student("Feda", "Economics")
    student_2 = Student("Sara", "Computer Science")

    print(f"student_1 name: {student_1.name}")
    print(f"student_1 major: {student_1.major}")
    print(f"student_2 name: {student_2.name}")
    print(f"student_2 major: {student_2.major}")


if __name__ == "__main__":
    main()

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
    print("OOP Minimum for Research")
 

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


# ==============================================================
# Instance methods
# ==============================================================

class Student:
    def __init__(self, name: str, major: str, gpa: float) -> None:
        self.name = name
        self.major = major
        self.gpa = gpa

    def describe(self) -> str:
        return f"{self.name} studies {self.major} and has GPA {self.gpa:.2f}"

    def is_honors(self) -> bool:
        return self.gpa >= 3.5


def main() -> None:
    section("1) Class with methods")

    student = Student("Feda", "Economics", 3.70)

    print(student.describe())
    print(f"Honors status: {student.is_honors()}")
    
    
    
# ==============================================================
#  __repr__ and research-style example
# ==============================================================

class Student:
    def __init__(self, name: str, major: str, gpa: float) -> None:
        self.name = name
        self.major = major
        self.gpa = gpa

    def describe(self) -> str:
        return f"{self.name} studies {self.major} and has GPA {self.gpa:.2f}"

    def is_honors(self) -> bool:
        return self.gpa >= 3.5

    def __repr__(self) -> str:
        # __repr__ defines how the object appears when printed.
        return f"Student(name='{self.name}', major='{self.major}', gpa={self.gpa:.2f})"


class DataPoint:
    def __init__(self, state: str, year: int, income: float) -> None:
        self.state = state
        self.year = year
        self.income = income

    def income_in_thousands(self) -> float:
        return self.income / 1000

    def __repr__(self) -> str:
        return f"DataPoint(state='{self.state}', year={self.year}, income={self.income:.2f})"


def main() -> None:
    section("1) Student class")

    student = Student("Feda", "Economics", 3.70)

    print(student)
    print(student.describe())
    print(f"Honors status: {student.is_honors()}")

    section("2) Research-style example")

    point = DataPoint("Kentucky", 2023, 42000)

    print(point)
    print(f"Income in thousands: {point.income_in_thousands():.2f}")


if __name__ == "__main__":
    main()



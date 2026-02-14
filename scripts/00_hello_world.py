"""
00_hello_world.py

Purpose:
--------
Introduce the most basic structure of a Python script.

Concepts covered:
- The main() function
- The __name__ == "__main__" pattern
- Basic input and output
- Clean script structure
"""


def main() -> None:
    """Main execution function."""

    print("Hello, World!")
    print("Welcome to Fundamentals of Python for Research.")

    name = input("What is your name? ")
    print(f"Nice to meet you, {name}.")

    print("\nYou are now ready to start building your Python skills.")


if __name__ == "__main__":
    main()

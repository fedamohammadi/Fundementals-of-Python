"""
Note:
Virtual environments are created in the terminal, not inside Python code.
This script is a simple CLI argument handling using sys.argv.
"""

import sys


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)



# ==============================================================
# sys.argv basics
# ==============================================================

def main() -> None:
    section("1) sys.argv basics")

    # sys.argv stores the command used to run the script.
    print("Raw sys.argv:")
    print(sys.argv)

    section("2) Script name")

    # The first item is always the script path/name.
    print(f"Script name: {sys.argv[0]}")


# ==============================================================
# Optional command line arguments
# ==============================================================
    section("2) Optional arguments")

    if len(sys.argv) > 1:
        name = sys.argv[1]
        print(f"Hello, {name}!")
    else:
        print("No name was passed on the command line.")

    section("3) Second argument example")

    if len(sys.argv) > 2:
        topic = sys.argv[2]
        print(f"Your topic is: {topic}")
    else:
        print("No second argument was provided.")





if __name__ == "__main__":
    main()
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



if __name__ == "__main__":
    main()
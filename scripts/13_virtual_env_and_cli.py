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



def main() -> None:
    section("Virtual Environments and CLI")
    print("This lesson introduces sys.argv and simple command line usage.")




if __name__ == "__main__":
    main()
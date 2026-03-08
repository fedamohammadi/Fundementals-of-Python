"""
Purpose:
--------
Learn how to read from and write to text and CSV files in Python.

Concepts covered (progressively):
- pathlib for safe file paths
- writing text files
- reading text files
- writing CSV files
- reading CSV files
- basic file existence checks

"""

from pathlib import Path
import csv


def section(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# Main ==========================

def main() -> None:
    section("File I/O and CSV")
    print("This file will introduce reading and writing files in Python.")




if __name__ == "__main__":
    main()

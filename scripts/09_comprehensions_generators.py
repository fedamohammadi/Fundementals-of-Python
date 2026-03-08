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


# ==============================================================
# Basic text file writing and reading
# ==============================================================

def main() -> None:
    section("1) Build file paths safely with pathlib")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)  # Create the folder if it does not exist.

    text_file = output_dir / "sample_notes.txt"
    print(f"Text file path: {text_file}")

    section("2) Write a text file")

    lines = [
        "Research Notes",
        "---------------",
        "Python is useful for reproducible research.",
        "Always document your workflow clearly."
    ]

    with open(text_file, mode="w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"Wrote text file: {text_file}")

    section("3) Read a text file")

    with open(text_file, mode="r", encoding="utf-8") as file:
        content = file.read()

    print("File content:")
    print(content)
   
# ==============================================================
# CSV writing and reading with csv module
# ==============================================================

def main() -> None:
    section("1) Build file paths safely with pathlib")

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    text_file = output_dir / "sample_notes.txt"
    csv_file = output_dir / "sample_data.csv"

    print(f"Text file path: {text_file}")
    print(f"CSV file path: {csv_file}")

    section("2) Write a text file")

    lines = [
        "Research Notes",
        "---------------",
        "Python is useful for reproducible research.",
        "Always document your workflow clearly."
    ]

    with open(text_file, mode="w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    print(f"Wrote text file: {text_file}")

    section("3) Write a CSV file")

    rows = [
        ["state", "year", "income"],
        ["Kentucky", 2023, 42000],
        ["Virginia", 2023, 51000],
        ["Ohio", 2023, 47000]
    ]

    # newline="" prevents blank lines in CSV files on Windows.
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print(f"Wrote CSV file: {csv_file}")

    section("4) Read the CSV file")

    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)  

if __name__ == "__main__":
    main()

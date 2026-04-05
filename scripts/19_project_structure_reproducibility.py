"""
Project Structure and Reproducibility
- organizing a project with pathlib
- managing config with JSON
- setting seeds for reproducible results
- reusable logging setup
- putting it all together
"""

from pathlib import Path


# ==============================================================
# Organizing a project with pathlib
# ==============================================================

# Define the project root relative to this file so paths work
# regardless of where the script is called from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR   = PROJECT_ROOT / "logs"


def demo_pathlib() -> None:
    # Build and inspect paths without hard-coding separators
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")

    # Check existence without raising an error
    print(f"data/ exists : {DATA_DIR.exists()}")

    # Safely create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output/ ready: {OUTPUT_DIR.exists()}")

    # Common path operations
    csv_path = DATA_DIR / "results.csv"
    print(f"Stem  : {csv_path.stem}")    # results
    print(f"Suffix: {csv_path.suffix}")  # .csv
    print(f"Parent: {csv_path.parent}")

    # Glob — find all .py files under scripts/
    scripts = sorted(PROJECT_ROOT.glob("scripts/*.py"))
    print(f"\nScripts found: {len(scripts)}")
    for s in scripts[:3]:
        print(f"  {s.name}")


def main() -> None:
    print("=" * 50)
    print("1. pathlib — portable project paths")
    print("=" * 50)
    demo_pathlib()


if __name__ == "__main__":
    main()

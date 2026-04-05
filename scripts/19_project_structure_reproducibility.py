"""
Project Structure and Reproducibility
- organizing a project with pathlib
- managing config with JSON
- setting seeds for reproducible results
- reusable logging setup
- putting it all together
"""

import json
import random
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


# ==============================================================
# Managing config with JSON
# ==============================================================

DEFAULT_CONFIG: dict = {
    "seed": 42,
    "n_samples": 1000,
    "output_format": "csv",
    "verbose": True,
}

CONFIG_PATH = OUTPUT_DIR / "config.json"


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def demo_config() -> None:
    # Save default config
    save_config(DEFAULT_CONFIG, CONFIG_PATH)

    # Load it back and inspect
    cfg = load_config(CONFIG_PATH)
    print(f"Loaded config: {cfg}")

    # Override a single value and re-save
    cfg["n_samples"] = 5000
    save_config(cfg, CONFIG_PATH)
    print(f"Updated n_samples: {load_config(CONFIG_PATH)['n_samples']}")

    # Clean up the demo file
    CONFIG_PATH.unlink(missing_ok=True)


# ==============================================================
# Seeds for reproducible results
# ==============================================================

def set_seeds(seed: int) -> None:
    """Set all relevant random seeds in one place."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def demo_seeds() -> None:
    # Without a seed — different every run
    sample_a = [random.randint(0, 100) for _ in range(5)]
    sample_b = [random.randint(0, 100) for _ in range(5)]
    print(f"No seed — run A: {sample_a}")
    print(f"No seed — run B: {sample_b}")

    # With a fixed seed — identical every run
    set_seeds(42)
    fixed_a = [random.randint(0, 100) for _ in range(5)]
    set_seeds(42)
    fixed_b = [random.randint(0, 100) for _ in range(5)]
    print(f"seed=42 — run A: {fixed_a}")
    print(f"seed=42 — run B: {fixed_b}")
    print(f"Identical: {fixed_a == fixed_b}")

    # NumPy reproducibility
    try:
        import numpy as np
        set_seeds(42)
        arr_a = np.random.normal(0, 1, 3).round(4)
        set_seeds(42)
        arr_b = np.random.normal(0, 1, 3).round(4)
        print(f"np seed=42 — run A: {arr_a}")
        print(f"np seed=42 — run B: {arr_b}")
    except ImportError:
        print("numpy not available — skipping numpy seed demo")


def main() -> None:
    print("=" * 50)
    print("1. pathlib — portable project paths")
    print("=" * 50)
    demo_pathlib()

    print()
    print("=" * 50)
    print("2. Config management with JSON")
    print("=" * 50)
    demo_config()

    print()
    print("=" * 50)
    print("3. Seeds and reproducibility")
    print("=" * 50)
    demo_seeds()


if __name__ == "__main__":
    main()

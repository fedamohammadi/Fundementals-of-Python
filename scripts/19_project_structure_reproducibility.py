"""
Project Structure and Reproducibility
- organizing a project with pathlib
- managing config with JSON
- setting seeds for reproducible results
- reusable logging setup
- putting it all together
"""

import json
import logging
import random
import sys
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


# ==============================================================
# Reusable logging setup
# ==============================================================

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Return a logger that writes to console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                            datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def demo_logging() -> None:
    log_path = LOGS_DIR / "demo.log"
    log = get_logger("project19", log_file=log_path)

    log.debug("Debug: fine-grained detail for development")
    log.info("Info: normal progress messages")
    log.warning("Warning: something unexpected but recoverable")
    log.error("Error: something went wrong")

    print(f"\nLog also written to: {log_path}")

    # Close handlers before deleting the file (required on Windows)
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)
    log_path.unlink(missing_ok=True)


# ==============================================================
# Putting it all together — a reproducible analysis run
# ==============================================================

def run_analysis(config: dict) -> dict:
    """Simulate a full reproducible pipeline using config values."""
    log = get_logger("run_analysis")

    set_seeds(config["seed"])
    log.info(f"Seed set to {config['seed']}")

    n = config["n_samples"]
    data = [random.gauss(0, 1) for _ in range(n)]
    mean = sum(data) / len(data)
    log.info(f"Generated {n} samples — mean: {mean:.4f}")

    results = {"n": n, "mean": round(mean, 6), "seed": config["seed"]}
    log.info(f"Results: {results}")
    return results


def demo_full_pipeline() -> None:
    cfg = DEFAULT_CONFIG.copy()

    # Save config so the run is fully documented
    cfg_path = OUTPUT_DIR / "run_config.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg_path)

    results = run_analysis(cfg)
    print(f"Final results: {results}")

    # Clean up
    cfg_path.unlink(missing_ok=True)

    # Close any open handlers to avoid Windows file locks
    log = logging.getLogger("run_analysis")
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


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

    print()
    print("=" * 50)
    print("4. Reusable logging setup")
    print("=" * 50)
    demo_logging()

    print()
    print("=" * 50)
    print("5. Full reproducible pipeline")
    print("=" * 50)
    demo_full_pipeline()


if __name__ == "__main__":
    main()

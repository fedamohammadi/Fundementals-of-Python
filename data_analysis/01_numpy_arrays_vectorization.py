"""
NumPy Arrays and Vectorization:
- Creating arrays (from lists, built-in generators, random)
- Array attributes: shape, dtype, ndim, size
- Indexing and slicing (1D, 2D, boolean masks)
- Arithmetic and broadcasting
- Aggregation and axis operations
- Vectorization vs. Python loops (speed comparison)
- Practical example: portfolio return statistics
"""

import time
import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# 1. Creating Arrays
# ==============================================================
# Every NumPy workflow starts with getting data into an ndarray.
# Four core creation patterns cover most real-world cases:
# from existing data, pre-filled arrays, sequences, and random draws.

def demo_creating_arrays() -> None:
    # From a Python list
    prices = np.array([102.5, 98.3, 105.7, 101.2, 99.8])
    print(f"\n  From list       : {prices}")

    # All-zeros and all-ones matrices (useful as placeholders)
    zeros = np.zeros((2, 3))
    ones  = np.ones((2, 3))
    print(f"\n  zeros(2×3):\n{zeros}")
    print(f"\n  ones(2×3) :\n{ones}")

    # arange: like Python range but returns an ndarray
    monthly = np.arange(0, 365, 30)
    print(f"\n  arange(0, 365, 30) — monthly checkpoints:\n  {monthly}")

    # linspace: n evenly-spaced points between start and stop (inclusive)
    discount_rates = np.linspace(0.01, 0.10, 10)
    print(f"\n  linspace(0.01, 0.10, 10) — discount rate grid:")
    print(f"  {discount_rates.round(3)}")

    # Random arrays — use a Generator for reproducibility
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(loc=0.001, scale=0.02, size=5)
    print(f"\n  Normal random (loc=0.001, scale=0.02, n=5):")
    print(f"  {returns.round(5)}")


# ==============================================================
# 2. Array Attributes and Reshaping
# ==============================================================
# Before manipulating an array, check its metadata: shape tells you
# dimensions, dtype controls memory and precision, and size gives
# the total element count. Reshape changes the view, not the data.

def demo_attributes_reshape() -> None:
    data = np.arange(12)
    print(f"\n  1D array: {data}")
    print(f"    shape  = {data.shape}")
    print(f"    ndim   = {data.ndim}")
    print(f"    size   = {data.size}")
    print(f"    dtype  = {data.dtype}")

    # Reshape to a 2D matrix (rows × cols must multiply to size)
    matrix = data.reshape(3, 4)
    print(f"\n  Reshaped to (3, 4):\n{matrix}")

    # -1 lets NumPy infer one dimension automatically
    col_vector = data.reshape(-1, 1)
    print(f"\n  Reshaped to (-1, 1) — shape={col_vector.shape}  (transposed for display):")
    print(f"  {col_vector.T}")

    # dtype controls precision and memory footprint
    float32_arr = np.array([1.0, 2.5, 3.7], dtype=np.float32)
    float64_arr = np.array([1.0, 2.5, 3.7], dtype=np.float64)
    print(f"\n  float32 bytes: {float32_arr.nbytes}  |  "
          f"float64 bytes: {float64_arr.nbytes}")

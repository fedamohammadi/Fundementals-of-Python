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


# ==============================================================
# 3. Indexing and Slicing
# ==============================================================
# NumPy extends Python's slice syntax to multiple dimensions.
# Boolean masks let you select by condition rather than position
# — the backbone of data filtering without loops.

def demo_indexing_slicing() -> None:
    rng   = np.random.default_rng(7)
    daily = rng.normal(0.0005, 0.015, size=10).round(4)
    print(f"\n  10 daily returns: {daily}")

    # Scalar access and slicing
    print(f"\n  First return        : {daily[0]}")
    print(f"  Last return         : {daily[-1]}")
    print(f"  Returns index 3-5   : {daily[3:6]}")
    print(f"  Every other return  : {daily[::2]}")

    # 2D indexing
    prices = np.array([
        [100, 101, 102],   # stock A
        [200, 198, 205],   # stock B
        [ 50,  51,  49],   # stock C
    ])
    print(f"\n  Price matrix (3 stocks × 3 days):\n{prices}")
    print(f"\n  Row 1 (stock B)     : {prices[1, :]}")
    print(f"  Column 2 (day 3)    : {prices[:, 2]}")
    print(f"  Sub-matrix [0:2, 1:3]:\n{prices[0:2, 1:3]}")

    # Boolean mask: select only positive returns
    pos_mask    = daily > 0
    pos_returns = daily[pos_mask]
    print(f"\n  Boolean mask (daily > 0): {pos_mask}")
    print(f"  Positive returns        : {pos_returns}")

    # Fancy indexing: select by a list of positions
    selected = daily[[0, 2, 4, 9]]
    print(f"\n  Fancy index [0,2,4,9]   : {selected}")


# ==============================================================
# 4. Arithmetic and Broadcasting
# ==============================================================
# NumPy operations apply element-wise to entire arrays with no loop.
# Broadcasting extends this to arrays of different — but compatible —
# shapes, aligning dimensions from the right and stretching size-1 axes.

def demo_arithmetic_broadcasting() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])

    print(f"\n  a = {a}")
    print(f"  b = {b}")
    print(f"\n  a + b  = {a + b}")
    print(f"  a * b  = {a * b}")
    print(f"  b / a  = {b / a}")
    print(f"  a ** 2 = {a ** 2}")

    # Scalar broadcasting: the scalar stretches to every element
    print(f"\n  a * 100 (scalar broadcast) = {a * 100}")

    # Shape broadcasting: (3, 1) stretches across (3, 3)
    weights = np.array([[0.5], [0.3], [0.2]])   # shape (3, 1)
    rets    = np.array([
        [ 0.01,  0.02, -0.01],
        [ 0.005, 0.01,  0.00],
        [-0.02,  0.03,  0.01],
    ])                                            # shape (3, 3)
    weighted = weights * rets                     # shape (3, 3)
    print(f"\n  Weight column:\n{weights.T}")
    print(f"\n  Returns matrix:\n{rets}")
    print(f"\n  Weighted returns (broadcast multiply):\n{weighted.round(4)}")

    # Universal functions (ufuncs): fast compiled math, element-wise
    rates = np.array([0.03, 0.05, 0.08, 0.10])
    print(f"\n  np.log1p(rates) — log continuous compounding: {np.log1p(rates).round(5)}")
    print(f"  np.exp(rates)   — growth factor e^rate       : {np.exp(rates).round(5)}")


# ==============================================================
# 5. Aggregation and Axis Operations
# ==============================================================
# Aggregation functions (sum, mean, std, min, max) collapse an axis.
# axis=0 collapses rows → per-column result.
# axis=1 collapses columns → per-row result.

def demo_aggregation() -> None:
    rng = np.random.default_rng(42)
    # 4 assets × 12 monthly returns
    returns = rng.normal(0.008, 0.04, size=(4, 12))

    print(f"\n  Monthly returns matrix — shape {returns.shape}  (4 assets × 12 months)")

    # Global aggregates (collapse everything)
    print(f"\n  Global mean    : {returns.mean():.5f}")
    print(f"  Global std     : {returns.std():.5f}")
    print(f"  Global min/max : {returns.min():.5f} / {returns.max():.5f}")

    # Per-asset: collapse months (axis=1)
    asset_means = returns.mean(axis=1)
    asset_stds  = returns.std(axis=1, ddof=1)
    print(f"\n  Per-asset mean (axis=1): {asset_means.round(5)}")
    print(f"  Per-asset std  (axis=1): {asset_stds.round(5)}")

    # Per-month: collapse assets (axis=0)
    month_means = returns.mean(axis=0)
    print(f"\n  Per-month mean (axis=0):\n  {month_means.round(5)}")

    # Cumulative product along time axis
    cum_returns  = np.cumprod(1 + returns[0])
    total_return = cum_returns[-1] - 1
    print(f"\n  Asset 0 — cumulative return over 12 months: {total_return * 100:.2f}%")


# ==============================================================
# 6. Vectorization vs. Python Loops
# ==============================================================
# A vectorized NumPy call runs in compiled C; a Python loop adds
# per-iteration interpreter overhead. On large arrays the speedup
# is 10–100x. The rule: replace any element-wise loop with a
# NumPy operator or ufunc.

def demo_vectorization() -> None:
    n = 1_000_000
    rng = np.random.default_rng(0)
    arr = rng.random(n)

    t0 = time.perf_counter()
    _loop = [x ** 2 for x in arr]
    loop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _vec = arr ** 2
    vec_time = time.perf_counter() - t0

    print(f"\n  Array size         : {n:,}")
    print(f"  Python loop time   : {loop_time:.4f} s")
    print(f"  NumPy vectorised   : {vec_time:.4f} s")
    print(f"  Speedup            : ~{loop_time / vec_time:.0f}×")

    # Practical pattern: simple returns via np.diff vs. a loop
    prices = np.array([100.0, 102.5, 98.3, 107.1, 105.0])
    returns_loop = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    returns_vec  = np.diff(prices) / prices[:-1]

    print(f"\n  Simple returns (loop) : {[round(r, 5) for r in returns_loop]}")
    print(f"  Simple returns (NumPy): {returns_vec.round(5)}")
    print(f"  Results identical     : {np.allclose(returns_loop, returns_vec)}")

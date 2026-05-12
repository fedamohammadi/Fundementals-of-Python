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

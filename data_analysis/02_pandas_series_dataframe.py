"""
Pandas Series and DataFrame:
- Creating a Series (from list, dict, scalar)
- Creating a DataFrame (from dict, list of dicts, NumPy array)
- Basic inspection: head, tail, shape, dtypes, describe, info
- Indexing and selection: [], .loc, .iloc, boolean masks
- Adding and dropping columns; renaming
- Applying functions: apply, map, vectorized string ops
- Practical example: student gradebook analysis
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

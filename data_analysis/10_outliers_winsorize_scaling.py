"""
Outliers, Winsorization, and Feature Scaling:
- Detecting outliers with the IQR method (Tukey's fences)
- Detecting outliers with the Z-score method
- Winsorizing: clipping extreme values to boundary percentiles
- Min-max scaling: normalising features to a [0, 1] range
- Standard scaling: zero mean and unit variance (z-score)
- Robust scaling: median-centred, IQR-normalised
- Practical example: cleaning and scaling a house price dataset
"""

import numpy as np
import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: house prices with injected outliers
# ==============================================================
# Each row is a property listing. Three extreme values were added
# deliberately — a $1 data-entry error, an implausible 18 000 sqft
# mansion, and a luxury sale — so the detection functions have
# clear targets while the rest of the data remains realistic.

def make_houses_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "sqft":      np.random.randint(800, 3_000, n),
        "bedrooms":  np.random.randint(1, 6, n),
        "age_years": np.random.randint(0, 50, n),
        "price":     np.random.randint(150_000, 600_000, n).astype(float),
    })
    df.loc[0,  "price"] = 1.0        # data-entry error
    df.loc[1,  "sqft"]  = 18_000     # implausible mansion
    df.loc[18, "price"] = 4_500_000  # luxury outlier
    return df

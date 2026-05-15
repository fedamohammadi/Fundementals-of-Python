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


# ==============================================================
# 1. IQR Method for Outlier Detection
# ==============================================================
# The interquartile range (IQR = Q3 - Q1) captures the spread of
# the middle 50 % of data. Tukey's fences mark anything below
# Q1 - 1.5×IQR or above Q3 + 1.5×IQR as an outlier. Using 3.0
# instead of 1.5 finds only "extreme" outliers — fewer false flags
# on genuinely heavy-tailed distributions.

def demo_iqr() -> None:
    df = make_houses_df()

    for col in ["price", "sqft"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags   = (df[col] < lo) | (df[col] > hi)
        print(f"\n  IQR outliers in '{col}':")
        print(f"  Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}  fence=[{lo:,.0f}, {hi:,.0f}]")
        print(f"  Flagged rows ({flags.sum()}):")
        if flags.any():
            print(df.loc[flags, ["sqft", "bedrooms", "age_years", col]].to_string())
        else:
            print("  (none)")

    # Combined flag: outlier in any numeric column
    outlier_mask = pd.Series(False, index=df.index)
    for col in ["price", "sqft", "bedrooms", "age_years"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        outlier_mask |= (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    print(f"\n  Rows flagged as outliers in any column: {outlier_mask.sum()} / {len(df)}")


# ==============================================================
# 2. Z-Score Method for Outlier Detection
# ==============================================================
# A z-score measures distance from the mean in units of standard
# deviation: z = (x - mean) / std. |z| > 3 is the standard
# threshold (~0.3 % of a normal distribution). The weakness: the
# mean and std are themselves pulled by the outliers being detected,
# making IQR the safer choice on skewed or contaminated data.

def demo_zscore() -> None:
    df = make_houses_df()

    for col in ["price", "sqft"]:
        mean, std = df[col].mean(), df[col].std()
        z         = (df[col] - mean) / std
        flags     = z.abs() > 3
        print(f"\n  Z-score outliers in '{col}' (|z| > 3):")
        print(f"  mean={mean:,.0f}  std={std:,.0f}")
        print(f"  Flagged rows ({flags.sum()}):")
        if flags.any():
            print(df.loc[flags, [col]].assign(z_score=z[flags].round(2)).to_string())
        else:
            print("  (none)")

    # Compare IQR vs z-score agreement on price
    price    = df["price"]
    q1, q3   = price.quantile([0.25, 0.75])
    iqr      = q3 - q1
    iqr_flag = (price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)
    z_flag   = ((price - price.mean()) / price.std()).abs() > 3
    print(f"\n  Price outlier agreement between methods:")
    print(f"  Both methods  : {(iqr_flag & z_flag).sum()}")
    print(f"  IQR only      : {(iqr_flag & ~z_flag).sum()}")
    print(f"  Z-score only  : {(z_flag & ~iqr_flag).sum()}")

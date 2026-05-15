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


# ==============================================================
# 3. Winsorizing: Clipping to Boundary Percentiles
# ==============================================================
# Winsorizing replaces values beyond a chosen percentile with the
# percentile boundary itself — no rows are dropped, so the sample
# size is preserved. Series.clip() is the direct implementation.
# Choosing tight percentiles (1/99) is conservative; 5/95 is common
# when the data is known to be noisy.

def demo_winsorize() -> None:
    df = make_houses_df().copy()

    # Clip price at the 5th and 95th percentiles
    lo_p, hi_p      = df["price"].quantile([0.05, 0.95])
    df["price_wins"] = df["price"].clip(lower=lo_p, upper=hi_p)

    comparison = df[["price", "price_wins"]].assign(
        changed=(df["price"] != df["price_wins"])
    )
    print(f"\n  Winsorized price (5th–95th percentile):")
    print(f"  Clip bounds: [{lo_p:,.0f}, {hi_p:,.0f}]")
    print(comparison.to_string())

    # Effect on summary statistics
    print(f"\n  Statistics before winsorizing:")
    print(df["price"].describe().round(0).to_string())
    print(f"\n  Statistics after winsorizing:")
    print(df["price_wins"].describe().round(0).to_string())

    # sqft: clip the mansion outlier at the 99th percentile
    lo_sq, hi_sq    = df["sqft"].quantile([0.01, 0.99])
    df["sqft_wins"]  = df["sqft"].clip(lower=lo_sq, upper=hi_sq)
    print(f"\n  sqft outlier clipped: {df['sqft'].max():,} -> {df['sqft_wins'].max():,}")


# ==============================================================
# 4. Min-Max Scaling (Normalisation)
# ==============================================================
# Min-max scaling maps each value linearly to [0, 1] using:
# x_scaled = (x - min) / (max - min). It preserves relative
# distances exactly but is sensitive to outliers — one extreme
# value shifts the min or max anchor, compressing everything else
# toward the centre. Winsorize first when outliers are present.

def demo_minmax_scaling() -> None:
    df = make_houses_df().copy()

    # Winsorize before scaling to avoid outlier distortion
    for col in ["price", "sqft"]:
        lo, hi     = df[col].quantile([0.05, 0.95])
        df[col]    = df[col].clip(lower=lo, upper=hi)

    for col in ["price", "sqft", "age_years"]:
        col_min, col_max = df[col].min(), df[col].max()
        df[f"{col}_mm"]  = (df[col] - col_min) / (col_max - col_min)

    print(f"\n  Min-max scaled columns (first 10 rows):")
    cols = ["price", "price_mm", "sqft", "sqft_mm", "age_years", "age_years_mm"]
    print(df[cols].head(10).round(3).to_string(index=False))

    # Verify the scaled range is exactly [0, 1]
    for col in ["price_mm", "sqft_mm", "age_years_mm"]:
        print(f"  {col}: min={df[col].min():.3f}  max={df[col].max():.3f}")


# ==============================================================
# 5. Standard Scaling (Standardisation)
# ==============================================================
# Standard scaling transforms values to z = (x - mean) / std,
# giving a column with mean ≈ 0 and std ≈ 1. Unlike min-max, it
# does not bound the output range — a genuine outlier will still
# appear far from zero, which makes outlier inspection easier after
# scaling without needing to know the original scale.

def demo_standard_scaling() -> None:
    df = make_houses_df().copy()

    for col in ["price", "sqft", "age_years"]:
        mean, std      = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std

    print(f"\n  Standard-scaled columns (first 10 rows):")
    cols = ["price", "price_z", "sqft", "sqft_z", "age_years", "age_years_z"]
    print(df[cols].head(10).round(3).to_string(index=False))

    # Verify mean ≈ 0 and std ≈ 1 for each scaled column
    for col in ["price_z", "sqft_z", "age_years_z"]:
        print(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}")


# ==============================================================
# 6. Robust Scaling
# ==============================================================
# Robust scaling uses the median and IQR:
# x_robust = (x - median) / IQR. Neither the median nor the IQR
# is pulled hard by extreme values, making this method suitable
# when outliers are present but cannot be removed. The centre is
# 0 for the median and the typical unit is one IQR.

def demo_robust_scaling() -> None:
    df = make_houses_df().copy()

    for col in ["price", "sqft", "age_years"]:
        median         = df[col].median()
        q1, q3         = df[col].quantile([0.25, 0.75])
        iqr            = q3 - q1
        df[f"{col}_r"] = (df[col] - median) / iqr

    print(f"\n  Robust-scaled columns (first 10 rows):")
    cols = ["price", "price_r", "sqft", "sqft_r", "age_years", "age_years_r"]
    print(df[cols].head(10).round(3).to_string(index=False))

    # Show how differently each method handles the luxury outlier at row 18
    price     = df["price"]
    lo, hi    = price.quantile([0.05, 0.95])
    price_mm  = (price.clip(lo, hi) - price.clip(lo, hi).min()) / (price.clip(lo, hi).max() - price.clip(lo, hi).min())
    price_z   = (price - price.mean()) / price.std()
    median    = price.median()
    iqr_val   = price.quantile(0.75) - price.quantile(0.25)
    price_r   = (price - median) / iqr_val
    print(f"\n  Price outlier at row 18 (price={df.loc[18, 'price']:,.0f}):")
    print(f"  Min-max scaled : {price_mm[18]:.3f}  (capped at 1 after winsorize)")
    print(f"  Standard scaled: {price_z[18]:.3f}")
    print(f"  Robust scaled  : {price_r[18]:.3f}")


# ==============================================================
# 7. Practical Example: Cleaning and Scaling a House Price Dataset
# ==============================================================
# Full pipeline: flag outliers with IQR, winsorize all features,
# then produce all three scaled versions of price side by side for
# direct comparison — the pattern used before feeding data to a model.

def demo_house_pipeline() -> None:
    df = make_houses_df()
    features = ["price", "sqft", "bedrooms", "age_years"]

    # Outlier report
    print(f"\n  --- Outlier report (IQR method) ---")
    for col in features:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags  = (df[col] < lo) | (df[col] > hi)
        print(f"  {col:<12}: {flags.sum()} outlier(s) flagged")

    # Winsorize all features at the 5th–95th percentile
    df_clean = df.copy()
    for col in features:
        lo_p, hi_p    = df_clean[col].quantile([0.05, 0.95])
        df_clean[col] = df_clean[col].clip(lower=lo_p, upper=hi_p)
    print(f"\n  --- After winsorizing (5th–95th) ---")
    print(df_clean[features].describe().round(0).to_string())

    # Apply all three scaling methods to price and compare
    col          = "price"
    col_min      = df_clean[col].min()
    col_max      = df_clean[col].max()
    mean, std    = df_clean[col].mean(), df_clean[col].std()
    median       = df_clean[col].median()
    iqr_val      = df_clean[col].quantile(0.75) - df_clean[col].quantile(0.25)

    result = pd.DataFrame({
        "price_raw"     : df[col],
        "price_wins"    : df_clean[col],
        "price_minmax"  : (df_clean[col] - col_min) / (col_max - col_min),
        "price_standard": (df_clean[col] - mean) / std,
        "price_robust"  : (df_clean[col] - median) / iqr_val,
    })
    print(f"\n  --- All scaled versions of price (first 10 rows) ---")
    print(result.head(10).round(3).to_string(index=False))

    print(f"\n  Dataset ready for modelling: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    section("1. IQR Method for Outlier Detection")
    demo_iqr()

    section("2. Z-Score Method for Outlier Detection")
    demo_zscore()

    section("3. Winsorizing: Clipping to Boundary Percentiles")
    demo_winsorize()

    section("4. Min-Max Scaling (Normalisation)")
    demo_minmax_scaling()

    section("5. Standard Scaling (Standardisation)")
    demo_standard_scaling()

    section("6. Robust Scaling")
    demo_robust_scaling()

    section("7. Practical Example: Cleaning and Scaling a House Price Dataset")
    demo_house_pipeline()


if __name__ == "__main__":
    main()

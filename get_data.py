from pathlib import Path
import statsmodels.api as sm


DATA_DIR = Path("data")


def save_rdataset(name: str, package: str, filename: str) -> None:
    """
    Download an R dataset via statsmodels and save it as CSV.
    """
    df = sm.datasets.get_rdataset(name, package).data
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(df)} cols={df.shape[1]}")


def save_statsmodels_dataset(dataset, filename: str) -> None:
    """
    Load a statsmodels built-in dataset and save it as CSV.
    """
    df = dataset.load_pandas().data
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(df)} cols={df.shape[1]}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # Wooldridge datasets (via get_rdataset)
    save_rdataset("wage1", "wooldridge", "wage1.csv")
    save_rdataset("mroz", "wooldridge", "mroz.csv")
    save_rdataset("card", "wooldridge", "card.csv")

    # Panel dataset
    save_statsmodels_dataset(sm.datasets.grunfeld, "grunfeld.csv")

    # Macro time series dataset
    save_statsmodels_dataset(sm.datasets.macrodata, "macrodata.csv")


if __name__ == "__main__":
    main()

# load raw data 
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/online_retail_II.csv")
PROCESSED_PATH = Path("data/processed/cleaned_online_retail.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Customer ID": "Int64"}, encoding="utf-8-sig")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["Customer ID", "InvoiceDate"])
    df = df.dropna(subset=["Description"])
    df = df[df["Quantity"] > 0]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Price"] > 0]
    df = df.drop_duplicates()
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {df.shape[0]} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd

PROCESSED_PATH = Path("data/processed/cleaned_online_retail.csv")
RFM_PATH = Path("data/processed/rfm.csv")


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Customer ID": "Int64"}, encoding="utf-8-sig")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    df = df.assign(Revenue=df["Quantity"] * df["Price"])

    rfm = (
        df.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    return rfm


def main() -> None:
    RFM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_data()
    rfm = build_rfm(df)
    rfm.to_csv(RFM_PATH, index=False)
    print(f"Built RFM for {rfm.shape[0]} customers -> {RFM_PATH}")
    print(rfm.head())


if __name__ == "__main__":
    main()
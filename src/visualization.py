
from pathlib import Path
import pandas as pd
import plotly.express as px

PROCESSED_PATH = Path("data/processed/cleaned_online_retail.csv")
OUTPUT_DIR = Path("reports")


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Customer ID": "Int64"}, encoding="utf-8-sig")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


def plot_purchasing_trends(df: pd.DataFrame):
    monthly = (
        df.set_index("InvoiceDate")
        .resample("ME")["Revenue"]  # use "M" if pandas < 2.2
        .sum()
        .reset_index()
    )
    return px.line(
        monthly,
        x="InvoiceDate",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend",
        labels={"InvoiceDate": "Month", "Revenue": "Revenue (£)"},
    )


def plot_top_products(df: pd.DataFrame, top_n: int = 10):
    top = (
        df.groupby("Description", as_index=False)["Revenue"]
        .sum()
        .nlargest(top_n, "Revenue")
        .sort_values("Revenue")
    )
    return px.bar(
        top,
        x="Revenue",
        y="Description",
        orientation="h",
        title=f"Top {top_n} Products by Revenue",
        labels={"Revenue": "Revenue (£)", "Description": ""},
    )


def plot_revenue_by_country(df: pd.DataFrame, top_n: int = 10):
    country = (
        df.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .nlargest(top_n, "Revenue")
        .sort_values("Revenue")
    )
    return px.bar(
        country,
        x="Revenue",
        y="Country",
        orientation="h",
        title=f"Top {top_n} Countries by Revenue",
        labels={"Revenue": "Revenue (£)", "Country": ""},
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    figures = {
        "purchasing_trends": plot_purchasing_trends(df),
        "top_products": plot_top_products(df),
        "revenue_by_country": plot_revenue_by_country(df),
    }

    for name, fig in figures.items():
        path = OUTPUT_DIR / f"{name}.html"
        fig.write_html(path)
        print(f"Saved {path}")
        fig.show()


if __name__ == "__main__":
    main()
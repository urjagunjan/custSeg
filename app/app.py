"""Interactive Streamlit dashboard for the RFM customer-segmentation project.

Run from the project root:

    streamlit run app/app.py

Provides:
  * portfolio-level Plotly charts (trends, top products, revenue by country)
  * segment summaries (sizes + centroids)
  * a Customer ID lookup showing the assigned segment and purchase history
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualization  # noqa: E402  (needs src/ on sys.path)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEGMENTS_PATH = DATA_DIR / "rfm_segments.csv"

FEATURES = ["Recency", "Frequency", "Monetary"]

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Loading transactions...")
def load_transactions() -> pd.DataFrame:
    """Load cleaned transactions with a computed Revenue column."""
    return visualization.load_data()


@st.cache_data(show_spinner="Loading segments...")
def load_segments() -> pd.DataFrame:
    """Load per-customer RFM values with cluster + segment labels."""
    return pd.read_csv(SEGMENTS_PATH, dtype={"Customer ID": "Int64"})


def render_overview(df: pd.DataFrame) -> None:
    """Render the three portfolio-level Plotly charts."""
    st.header("Portfolio Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(visualization.plot_purchasing_trends(df), use_container_width=True)
    with col2:
        st.plotly_chart(visualization.plot_revenue_by_country(df), use_container_width=True)
    st.plotly_chart(visualization.plot_top_products(df), use_container_width=True)


def render_segment_summary(segments: pd.DataFrame) -> None:
    """Render segment sizes and centroid table."""
    st.header("Segment Summary")
    counts = (
        segments["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )
    st.plotly_chart(
        px.bar(counts, x="Segment", y="Customers", title="Customers per Segment"),
        use_container_width=True,
    )
    st.dataframe(
        segments.groupby("Segment")[FEATURES]
        .mean()
        .round(2)
        .sort_values("Monetary", ascending=False)
    )


def render_customer_lookup(segments: pd.DataFrame, transactions: pd.DataFrame) -> None:
    """Let a stakeholder pick a Customer ID and see their segment + history."""
    st.header("Customer Lookup")
    customer_id = st.number_input(
        "Customer ID",
        min_value=int(segments["Customer ID"].min()),
        step=1,
        value=int(segments["Customer ID"].min()),
    )

    match = segments[segments["Customer ID"] == customer_id]
    if match.empty:
        st.warning(f"No customer found with ID {customer_id}.")
        return

    row = match.iloc[0]
    st.success(f"Customer {customer_id} -> {row['Segment']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Recency (days)", f"{row['Recency']:.0f}")
    c2.metric("Frequency (invoices)", f"{row['Frequency']:.0f}")
    c3.metric("Monetary (GBP)", f"{row['Monetary']:.2f}")

    history = transactions[transactions["Customer ID"] == customer_id].sort_values("InvoiceDate")
    if history.empty:
        st.info("No transaction history found for this customer.")
        return

    spend = history.set_index("InvoiceDate").resample("ME")["Revenue"].sum().reset_index()
    st.plotly_chart(
        px.line(spend, x="InvoiceDate", y="Revenue", markers=True, title="Monthly Spend"),
        use_container_width=True,
    )
    st.dataframe(
        history[["Invoice", "InvoiceDate", "Description", "Quantity", "Price", "Revenue"]]
    )


def main() -> None:
    st.title("Customer Segmentation Dashboard")
    transactions = load_transactions()
    segments = load_segments()

    page = st.sidebar.radio("View", ["Overview", "Segments", "Customer Lookup"])
    if page == "Overview":
        render_overview(transactions)
    elif page == "Segments":
        render_segment_summary(segments)
    else:
        render_customer_lookup(segments, transactions)


if __name__ == "__main__":
    main()
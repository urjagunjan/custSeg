# custSeg

An end-to-end Machine Learning pipeline for e-commerce customer segmentation using RFM analysis and K-Means clustering, featuring an interactive Streamlit dashboard.

## Overview

`custSeg` cleans raw online-retail transactions, engineers Recency/Frequency/Monetary (RFM) features per customer, standardizes them, clusters customers with K-Means, assigns business-friendly segment names, and visualizes everything in a Streamlit app.

## Project Structure

```
custSeg/
|-- app/
|   `-- app.py                 # Streamlit dashboard
|-- data/
|   |-- raw/                   # raw transactions (online_retail_II.csv)
|   `-- processed/             # pipeline artifacts (csv)
|-- src/
|   |-- data_cleaning.py       # ingest + clean raw data
|   |-- rfm.py                 # build RFM features
|   |-- preprocess_rfm.py      # log1p + StandardScaler
|   |-- segmentation.py        # K-Means + elbow/silhouette
|   |-- segment_naming.py      # map clusters -> business segments
|   |-- visualization.py       # Plotly charts
|   `-- rfm_pipeline.py        # single entry point (runs all stages)
|-- requirements.txt
`-- README.md
```

## Setup

Requirements: Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### 1. Run the full pipeline

From the project root:

```powershell
python src/rfm_pipeline.py
```

This produces, in order: `cleaned_online_retail.csv`, `rfm.csv`, `rfm_scaled.csv`,
`rfm_clustered.csv`, and `rfm_segments.csv` under `data/processed/`.

Force a cluster count:

```powershell
python src/rfm_pipeline.py --stage train --k 4
python src/segment_naming.py
```

### 2. Launch the dashboard

```powershell
streamlit run app/app.py
```

## Methodology

1. **Cleaning** - drop missing Customer IDs/descriptions, remove cancellations, keep `Quantity > 0` and `Price > 0`, de-duplicate.
2. **RFM features** - Recency = days since last purchase (snapshot = max date + 1); Frequency = distinct invoices; Monetary = sum(Quantity x Price).
3. **Transformation** - `log1p` to tame right-skew, then `StandardScaler` so no feature dominates distance.
4. **Clustering** - K-Means; k chosen via the Elbow method (inertia) and Silhouette score.
5. **Naming** - clusters labeled against dataset medians (Champions, Loyal, At-Risk, Hibernating, ...).

## Model Limitations

- **K-Means assumes spherical, equally sized clusters** and is sensitive to initialization and outliers; `n_init=10` and `random_state=42` mitigate but do not remove this.
- **RFM collapses behavior into three numbers**, ignoring product mix, seasonality, returns, and channel.
- **Stationary snapshot**: the model is a point-in-time view; customers migrate between segments and the model should be re-fit periodically.
- **No ground-truth labels** - segment count and names are heuristic, validated only by elbow/silhouette and business judgment.
- **`log1p` + scaling are fit on the full dataset**, which is fine for descriptive analysis but leaks if used for a predictive train/test split.
- **New/unseen customers** are not scored by the dashboard unless the scaler + K-Means model are persisted and re-applied.

## Business Insights

> Replace the figures below with values from `data/processed/rfm_segments.csv`.

- **Champions** (high R/F/M) are the smallest group but drive a disproportionate share of revenue - protect with loyalty perks and early access.
- **Loyal Customers** buy often at moderate value - upsell and cross-sell to move them toward Champions.
- **At-Risk** customers were valuable but have gone quiet - target with win-back campaigns before they lapse.
- **Hibernating / Lost** (low R/F/M) make up the largest group but contribute little - use low-cost reactivation emails; avoid heavy spend.
- **New Customers** are recent but low frequency - nurture with onboarding flows to build a buying habit.

Actionable summary: concentrate retention budget on Champions and At-Risk, use automation for Hibernating, and design onboarding to convert New Customers into Loyal ones.

## Future Work

- Persist the scaler + K-Means model (`joblib`) to score new customers in real time.
- Add CLV prediction and segment-transition tracking over time.
- Validate segment stability across time windows (e.g. bootstrap or holdout months).

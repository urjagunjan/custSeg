from pathlib import Path

import pandas as pd

CLUSTERED_PATH = Path("data/processed/rfm_clustered.csv")
OUTPUT_PATH = Path("data/processed/rfm_segments.csv")

FEATURES = ["Recency", "Frequency", "Monetary"]


def load_clustered(path: Path = CLUSTERED_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Customer ID": "Int64", "Cluster": "Int64"})


def cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.groupby("Cluster")[FEATURES].mean().round(2)
    profile["Customers"] = df["Cluster"].value_counts().sort_index()
    profile["Share %"] = (profile["Customers"] / len(df) * 100).round(1)
    return profile


def assign_segment_names(df: pd.DataFrame, profile: pd.DataFrame) -> dict:
    med = df[FEATURES].median()
    names = {}
    for cluster, row in profile.iterrows():
        recent = row["Recency"] <= med["Recency"]
        frequent = row["Frequency"] >= med["Frequency"]
        spend = row["Monetary"] >= med["Monetary"]

        if recent and frequent and spend:
            name = "Champions"
        elif recent and frequent and not spend:
            name = "Loyal Customers"
        elif recent and not frequent and spend:
            name = "Big Spenders"
        elif recent and not frequent and not spend:
            name = "New Customers"
        elif not recent and frequent and spend:
            name = "At-Risk (High Value)"
        elif not recent and frequent and not spend:
            name = "At-Risk (Low Value)"
        elif not recent and not frequent and spend:
            name = "Hibernating (High Value)"
        else:
            name = "Hibernating / Lost"
        names[cluster] = name
    return names


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_clustered()

    profile = cluster_profile(df)
    print("=== Cluster centroids (original RFM units) ===")
    print(profile)

    names = assign_segment_names(df, profile)
    df["Segment"] = df["Cluster"].map(names)
    df.to_csv(OUTPUT_PATH, index=False)

    seg_profile = (
        df.groupby("Segment")[FEATURES]
        .mean()
        .round(2)
        .assign(Customers=df["Segment"].value_counts())
        .sort_values("Monetary", ascending=False)
    )
    print("\n=== Segment profile ===")
    print(seg_profile)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
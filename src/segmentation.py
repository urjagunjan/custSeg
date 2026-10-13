from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SCALED_PATH = Path("data/processed/rfm_scaled.csv")
RFM_PATH = Path("data/processed/rfm.csv")
OUTPUT_PATH = Path("data/processed/rfm_clustered.csv")
PLOT_PATH = Path("reports/k_selection.png")

FEATURES = ["Recency", "Frequency", "Monetary"]
K_RANGE = range(2, 11)
RANDOM_STATE = 42


def load_scaled(path: Path = SCALED_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Customer ID": "Int64"})


def evaluate_k(X, k_range=K_RANGE):
    ks, inertia, silhouettes = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        ks.append(k)
        inertia.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return ks, inertia, silhouettes


def plot_metrics(ks, inertia, silhouettes, path: Path = PLOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ks, inertia, marker="o")
    ax1.set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia")
    ax2.plot(ks, silhouettes, marker="s", color="tab:red")
    ax2.set(title="Silhouette Score", xlabel="Number of clusters (k)", ylabel="Silhouette")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(k: int | None = None) -> None:
    scaled = load_scaled()
    X = scaled[FEATURES].to_numpy()

    ks, inertia, silhouettes = evaluate_k(X)
    plot_metrics(ks, inertia, silhouettes)

    for k, i, s in zip(ks, inertia, silhouettes):
        print(f"k={k:2d}  inertia={i:12.1f}  silhouette={s:.3f}")

    best_k = k if k is not None else ks[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette score: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    out = scaled.copy()
    out["Cluster"] = km.fit_predict(X)

    rfm = pd.read_csv(RFM_PATH, dtype={"Customer ID": "Int64"})
    merged = rfm.merge(out[["Customer ID", "Cluster"]], on="Customer ID")
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved labels for {merged.shape[0]} customers -> {OUTPUT_PATH}")
    print("\nCluster profile (raw means):")
    print(merged.groupby("Cluster")[FEATURES].mean().round(2))


if __name__ == "__main__":
    main()
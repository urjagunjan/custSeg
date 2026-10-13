from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RFM_PATH = Path("data/processed/rfm.csv")
SCALED_PATH = Path("data/processed/rfm_scaled.csv")

FEATURES = ["Recency", "Frequency", "Monetary"]


def load_rfm(path: Path = RFM_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Customer ID": "Int64"})


def transform(rfm: pd.DataFrame) -> pd.DataFrame:
    log_df = np.log1p(rfm[FEATURES])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_df)
    out = pd.DataFrame(scaled, columns=FEATURES)
    out.insert(0, "Customer ID", rfm["Customer ID"].values)
    return out


def main() -> None:
    SCALED_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm = load_rfm()
    scaled = transform(rfm)
    scaled.to_csv(SCALED_PATH, index=False)
    print(f"Scaled {scaled.shape[0]} customers -> {SCALED_PATH}")
    print(scaled[FEATURES].describe().round(3))


if __name__ == "__main__":
    main()
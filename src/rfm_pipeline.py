"""End-to-end RFM customer-segmentation pipeline (ingest -> business segments).

This is the single, documented entry point that orchestrates the stages which
used to live in separate scripts / notebooks. Each stage is a thin wrapper
around one focused module in ``src/``:

    1. ingest     data_cleaning.main()      raw transactions -> cleaned transactions
    2. features   rfm.main()                cleaned -> per-customer Recency/Frequency/Monetary
    3. transform  preprocess_rfm.main()     RFM -> log1p + StandardScaler
    4. train      segmentation.main(k)      scaled RFM -> cluster labels
    5. label      segment_naming.main()     clusters -> business-friendly segment names

Run the whole pipeline from the project root::

    python src/rfm_pipeline.py

Run a single stage (optionally forcing k in the training stage)::

    python src/rfm_pipeline.py --stage train --k 4
"""
from __future__ import annotations

import argparse
import logging
import time

import data_cleaning
import preprocess_rfm
import rfm as rfm_features
import segment_naming
import segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rfm_pipeline")

STAGES = ("ingest", "features", "transform", "train", "label", "all")


def run_ingest() -> None:
    """Clean raw transactions -> data/processed/cleaned_online_retail.csv."""
    data_cleaning.main()


def run_features() -> None:
    """Build per-customer RFM -> data/processed/rfm.csv."""
    rfm_features.main()


def run_transform() -> None:
    """Log-transform + standardise RFM -> data/processed/rfm_scaled.csv."""
    preprocess_rfm.main()


def run_train(k: int | None = None) -> None:
    """Fit K-Means, evaluate k, and write data/processed/rfm_clustered.csv."""
    segmentation.main(k=k)


def run_label() -> None:
    """Attach business segment names -> data/processed/rfm_segments.csv."""
    segment_naming.main()


STAGE_FUNCS = {
    "ingest": run_ingest,
    "features": run_features,
    "transform": run_transform,
    "label": run_label,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Cluster count for the 'train' stage; auto-selected when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = (
        ("ingest", "features", "transform", "train", "label")
        if args.stage == "all"
        else (args.stage,)
    )

    for stage in stages:
        LOGGER.info("Running stage: %s", stage)
        started = time.perf_counter()
        if stage == "train":
            run_train(k=args.k)
        else:
            STAGE_FUNCS[stage]()
        LOGGER.info("Stage '%s' finished in %.1fs", stage, time.perf_counter() - started)


if __name__ == "__main__":
    main()
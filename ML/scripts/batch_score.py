"""Offline batch scoring - the throughput half of the hybrid serving
pattern described in DESIGN_DOC.md.

Scores every row in an input CSV in one process (no HTTP round-trips),
reusing the exact same model bundle and feature code as the online API.
This is the pattern a nightly job producing a retention-outreach list
would use: no one is waiting on any single row, so batch throughput
matters more than per-request latency.

Usage:
    python scripts/batch_score.py --input data/processed/training_data.csv --output artifacts/eval/batch_scores.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve
from src.features import prepare_xy
from src.serving.model_loader import load_production_model


def main():
    parser = argparse.ArgumentParser(description="Batch-score a CSV of customer records.")
    parser.add_argument("--input", type=str, default="data/processed/training_data.csv")
    parser.add_argument("--output", type=str, default="artifacts/eval/batch_scores.csv")
    args = parser.parse_args()

    cfg = load_config()
    bundle = load_production_model(cfg)

    input_path = resolve(args.input)
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    start = time.perf_counter()
    X, _ = prepare_xy(df, cfg, bundle.thresholds)
    proba = bundle.pipeline.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - start

    out = pd.DataFrame(
        {
            "customerID": df.get("customerID", pd.RangeIndex(len(df))),
            "churn_probability": proba.round(4),
            "churn_prediction": [cfg["target"]["positive_label"] if p >= 0.5 else "No" for p in proba],
            "model_version": bundle.version,
        }
    )
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    rows_per_sec = len(df) / elapsed if elapsed > 0 else float("inf")
    print(f"Scored {len(df)} rows in {elapsed:.3f}s ({rows_per_sec:.1f} rows/sec)")
    print(f"Wrote scores -> {output_path}")


if __name__ == "__main__":
    main()

"""Micro-batch data ingestion.

Simulates the kind of daily-drop ingestion a real pipeline would run on a
schedule: pick up new files from an incoming directory, validate their
schema, append them to the accumulating training data table, and log what
happened. Idempotent -- files already recorded in the ingestion log are
skipped on re-run, so this can be invoked repeatedly (e.g. from cron)
without double-counting rows.

Usage:
    python src/data_ingestion.py
    python src/data_ingestion.py --incoming-dir data/incoming --processed-path data/processed/training_data.csv
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve


def _load_ingested_files(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    ingested = set()
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ingested.add(json.loads(line)["source_file"])
    return ingested


def _append_log(log_path: Path, entry: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _validate_schema(df: pd.DataFrame, required_columns: list, source_name: str):
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name}: missing required columns {sorted(missing)} - rejecting file"
        )


def _rel_or_abs(path: Path) -> str:
    """Project-relative path when possible (readable log entries); falls
    back to an absolute path for files outside the project root, e.g. in
    tests using a tmp_path fixture.
    """
    try:
        return str(path.relative_to(resolve()))
    except ValueError:
        return str(path.resolve())


def ingest(
    cfg: dict,
    incoming_dir: Path = None,
    processed_path: Path = None,
    seed_path: Path = None,
    log_path: Path = None,
):
    incoming_dir = incoming_dir or resolve(cfg["paths"]["incoming_dir"])
    processed_path = processed_path or resolve(cfg["paths"]["processed_data"])
    log_path = log_path or resolve(cfg["paths"]["ingestion_log"])
    seed_path = seed_path or resolve(cfg["paths"]["raw_dir"], "historical_seed.csv")
    required_columns = cfg["required_columns"]

    already_ingested = _load_ingested_files(log_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # Bootstrap: seed the processed table from the historical file on first run.
    if not processed_path.exists() and seed_path.exists():
        seed_name = _rel_or_abs(seed_path)
        if seed_name not in already_ingested:
            seed_df = pd.read_csv(seed_path)
            _validate_schema(seed_df, required_columns, seed_name)
            seed_df.to_csv(processed_path, index=False)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_file": seed_name,
                "rows_ingested": len(seed_df),
                "cumulative_rows": len(seed_df),
                "status": "seeded",
            }
            _append_log(log_path, entry)
            already_ingested.add(seed_name)
            results.append(entry)
            print(f"[seed] {seed_name}: {len(seed_df)} rows -> {processed_path}")

    cumulative = 0
    if processed_path.exists():
        cumulative = sum(1 for _ in open(processed_path)) - 1  # minus header

    incoming_files = sorted(Path(incoming_dir).glob("*.csv"))
    for path in incoming_files:
        rel_name = _rel_or_abs(path)
        if rel_name in already_ingested:
            continue

        df = pd.read_csv(path)
        _validate_schema(df, required_columns, rel_name)

        df.to_csv(processed_path, mode="a", header=False, index=False)
        cumulative += len(df)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_file": rel_name,
            "rows_ingested": len(df),
            "cumulative_rows": cumulative,
            "status": "ingested",
        }
        _append_log(log_path, entry)
        results.append(entry)
        print(f"[ingest] {rel_name}: +{len(df)} rows (cumulative {cumulative}) -> {processed_path}")

    if not results:
        print("No new files to ingest.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ingest new batch files into the training data table.")
    parser.add_argument("--incoming-dir", type=str, default=None)
    parser.add_argument("--processed-path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    ingest(
        cfg,
        incoming_dir=Path(args.incoming_dir) if args.incoming_dir else None,
        processed_path=Path(args.processed_path) if args.processed_path else None,
    )


if __name__ == "__main__":
    main()

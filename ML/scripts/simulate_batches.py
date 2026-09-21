"""One-off setup script: splits the full Telco churn CSV into a historical
"training seed" and 5 dated micro-batch files under data/incoming/, to
stand in for daily upstream data drops in a real pipeline.

This is NOT part of the repeatable pipeline (data_ingestion.py is) -- it
just creates fixtures so ingestion/monitoring have something realistic to
run against. day_04 has MonthlyCharges and contract mix deliberately
shifted so the drift checker in src/monitoring.py has a real signal to
catch.

Usage: python scripts/simulate_batches.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / "data" / "raw" / "telco_churn_full.csv"
SEED_PATH = ROOT / "data" / "raw" / "historical_seed.csv"
INCOMING_DIR = ROOT / "data" / "incoming"

RANDOM_STATE = 42
N_DAYS = 5
DRIFTED_DAY = 4  # 1-indexed day that gets injected drift
SEED_FRACTION = 0.80


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate an upstream pricing change + a shift toward month-to-month
    contracts, the kind of real-world change a drift monitor should catch.
    """
    df = df.copy()
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.25).round(2)
    n_flip = int(len(df) * 0.35)
    flip_idx = df.sample(n=n_flip, random_state=RANDOM_STATE).index
    df.loc[flip_idx, "Contract"] = "Month-to-month"
    return df


def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing raw dataset at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    n_seed = int(len(df) * SEED_FRACTION)
    seed_df = df.iloc[:n_seed].reset_index(drop=True)
    remaining = df.iloc[n_seed:].reset_index(drop=True)

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(SEED_PATH, index=False)
    print(f"Wrote historical seed: {len(seed_df)} rows -> {SEED_PATH}")

    day_index_chunks = np.array_split(np.arange(len(remaining)), N_DAYS)
    dates = pd.date_range("2026-08-01", periods=N_DAYS, freq="D")

    for i, (idx, date) in enumerate(zip(day_index_chunks, dates), start=1):
        chunk = remaining.iloc[idx].reset_index(drop=True)
        if i == DRIFTED_DAY:
            chunk = inject_drift(chunk)
            note = " (drift injected: +25% MonthlyCharges, contract mix shift)"
        else:
            note = ""
        out_path = INCOMING_DIR / f"day_{i:02d}_{date.date()}.csv"
        chunk.to_csv(out_path, index=False)
        print(f"Wrote {out_path.name}: {len(chunk)} rows{note}")


if __name__ == "__main__":
    main()

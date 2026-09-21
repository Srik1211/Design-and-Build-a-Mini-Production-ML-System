"""Latency/throughput measurement for the online /predict endpoint.

Sends N sequential requests built from real rows in the processed
dataset (so payloads are realistic, not synthetic edge cases) and reports
avg/p50/p95/max latency plus overall throughput.

Usage:
    uvicorn src.serving.app:app --port 8000 &
    python scripts/load_test.py --n 200 --url http://127.0.0.1:8000/predict
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve

RAW_INPUT_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def main():
    parser = argparse.ArgumentParser(description="Measure /predict latency and throughput.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--data", type=str, default="data/processed/training_data.csv")
    parser.add_argument("--output", type=str, default="artifacts/eval/latency_report.json")
    args = parser.parse_args()

    cfg = load_config()
    df = pd.read_csv(resolve(args.data))
    sample = df.sample(n=min(args.n, len(df)), random_state=42).reset_index(drop=True)

    latencies_ms = []
    errors = 0
    with httpx.Client(timeout=10.0) as client:
        overall_start = time.perf_counter()
        for _, row in sample.iterrows():
            payload = {col: row[col] for col in RAW_INPUT_COLUMNS}
            start = time.perf_counter()
            resp = client.post(args.url, json=payload)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            if resp.status_code != 200:
                errors += 1
        overall_elapsed = time.perf_counter() - overall_start

    latencies = np.array(latencies_ms)
    report = {
        "n_requests": len(sample),
        "errors": errors,
        "avg_latency_ms": round(float(latencies.mean()), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "max_latency_ms": round(float(latencies.max()), 2),
        "total_elapsed_s": round(overall_elapsed, 3),
        "throughput_req_per_sec": round(len(sample) / overall_elapsed, 2),
    }

    print(json.dumps(report, indent=2))

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report -> {output_path}")


if __name__ == "__main__":
    main()

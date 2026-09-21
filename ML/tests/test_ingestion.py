import json

import pandas as pd

from src.config import load_config
from src.data_ingestion import ingest

CFG = load_config()
REQUIRED_COLUMNS = CFG["required_columns"]


def _make_df(n: int, start_id: int = 0) -> pd.DataFrame:
    data = {col: ["Yes"] * n for col in REQUIRED_COLUMNS}
    data["customerID"] = [f"CUST-{start_id + i}" for i in range(n)]
    data["SeniorCitizen"] = [0] * n
    data["tenure"] = list(range(n))
    data["MonthlyCharges"] = [50.0] * n
    data["TotalCharges"] = [100.0] * n
    data["Churn"] = ["No"] * n
    return pd.DataFrame(data)


def _setup(tmp_path):
    raw_dir = tmp_path / "raw"
    incoming_dir = tmp_path / "incoming"
    raw_dir.mkdir()
    incoming_dir.mkdir()

    seed_path = raw_dir / "historical_seed.csv"
    _make_df(10).to_csv(seed_path, index=False)

    processed_path = tmp_path / "processed" / "training_data.csv"
    log_path = tmp_path / "ingestion_log.jsonl"
    return incoming_dir, processed_path, seed_path, log_path


def test_ingest_seeds_from_historical_file(tmp_path):
    incoming_dir, processed_path, seed_path, log_path = _setup(tmp_path)

    results = ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)

    assert processed_path.exists()
    assert len(pd.read_csv(processed_path)) == 10
    assert len(results) == 1
    assert results[0]["status"] == "seeded"
    assert results[0]["rows_ingested"] == 10


def test_ingest_appends_new_incoming_files(tmp_path):
    incoming_dir, processed_path, seed_path, log_path = _setup(tmp_path)
    ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)

    day1 = incoming_dir / "day_01.csv"
    _make_df(5, start_id=100).to_csv(day1, index=False)

    results = ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)

    assert len(results) == 1
    assert results[0]["rows_ingested"] == 5
    assert results[0]["cumulative_rows"] == 15
    assert len(pd.read_csv(processed_path)) == 15


def test_ingest_is_idempotent_on_rerun(tmp_path):
    incoming_dir, processed_path, seed_path, log_path = _setup(tmp_path)
    day1 = incoming_dir / "day_01.csv"
    _make_df(5, start_id=100).to_csv(day1, index=False)

    ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)
    results_rerun = ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)

    assert results_rerun == []
    assert len(pd.read_csv(processed_path)) == 15  # unchanged, no duplicate rows


def test_ingest_logs_entries(tmp_path):
    incoming_dir, processed_path, seed_path, log_path = _setup(tmp_path)
    ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["rows_ingested"] == 10
    assert "timestamp" in entry


def test_ingest_rejects_file_missing_required_columns(tmp_path):
    incoming_dir, processed_path, seed_path, log_path = _setup(tmp_path)
    ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)

    bad_df = _make_df(3, start_id=200).drop(columns=["Contract"])
    (incoming_dir / "day_bad.csv").write_text(bad_df.to_csv(index=False))

    try:
        ingest(CFG, incoming_dir=incoming_dir, processed_path=processed_path, seed_path=seed_path, log_path=log_path)
        assert False, "expected ValueError for missing required column"
    except ValueError as exc:
        assert "Contract" in str(exc)

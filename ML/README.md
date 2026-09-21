# Churn Prediction — Production ML Pipeline

Binary classification: predict whether a telecom customer will churn.
Built to demonstrate a full, if intentionally minimal, production ML
lifecycle — ingestion, training, serving, and monitoring — rather than to
chase leaderboard accuracy. See `DESIGN_DOC.md` for the full write-up and
`architecture_diagram.png` for the system diagram.

## Setup

```bash
conda activate {env_name}
pip install -r requirements.txt   # fastapi/uvicorn/pydantic/pytest via conda-forge if pip is restricted
```

Dataset: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(~7,043 rows), already downloaded to `data/raw/telco_churn_full.csv`.

## Repository layout

```
configs/config.yaml       central paths, feature lists, thresholds
data/                     raw, incoming (simulated daily drops), processed (accumulated table)
src/features.py           SHARED feature engineering (train + serve both import this)
src/schema.py             pydantic request/response models
src/data_ingestion.py     micro-batch ingestion CLI
src/train.py              training + offline evaluation + promotion decision
src/monitoring.py         data quality, drift checks, retraining trigger
src/serving/app.py        FastAPI online inference
scripts/                  one-off/support scripts (see below)
models/                   registry.json + baseline/candidate model artifacts
artifacts/eval/           metrics, comparison report, latency report
monitoring/               drift check reports
tests/                    pytest suite
notebooks/                executed walkthrough notebook (see below)
```

## Notebook walkthrough

`notebooks/pipeline_walkthrough.ipynb` runs every stage below end-to-end —
ingestion, feature engineering, training/evaluation, online serving, batch
scoring, a live latency measurement, and drift/monitoring checks — by
importing or shelling out to the real files in `src/` and `scripts/`
(nothing is reimplemented). It is already executed and saved with its
outputs, so opening it is enough to see current results without running
anything; re-run it top to bottom (`jupyter nbconvert --to notebook
--execute --inplace notebooks/pipeline_walkthrough.ipynb
--ExecutePreprocessor.kernel_name=bits_clustering`) to regenerate them.
The notebook includes a note on why its training section reconstructs the
pre-drift data snapshot rather than calling `src/train.py` directly — see
the ordering caveat below.

## Reproducing the pipeline end-to-end

The commands below reproduce this repo's current state from scratch. Run
everything from `ML/` with `bits_clustering` active.

```bash
# 1. One-time fixture setup: splits the raw CSV into a historical seed +
#    5 simulated daily batches (day_04 has deliberately injected drift).
python scripts/simulate_batches.py

# 2. Ingest the seed + arriving daily files into the training data table.
#    Idempotent - safe to re-run; already-ingested files are skipped.
python src/data_ingestion.py

# 3. Train baseline (LogisticRegression) vs candidate (RandomForest),
#    apply the promotion rule, write models/ + artifacts/eval/.
python src/train.py

# 4. Run the test suite.
pytest tests -q

# 5. Start the online API.
uvicorn src.serving.app:app --reload --port 8000
# in another shell:
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @- <<'EOF'
{"customerID":"0000-DEMO","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
 "tenure":5,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
 "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
 "StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
 "PaymentMethod":"Electronic check","MonthlyCharges":85.5,"TotalCharges":"427.5"}
EOF

# 6. Measure latency/throughput against the running API.
python scripts/load_test.py --n 200

# 7. Offline batch scoring (the other half of the hybrid serving pattern).
python scripts/batch_score.py

# 8. Data quality + drift checks against a "recent batch" file.
python src/monitoring.py --recent-batch data/incoming/day_04_2026-08-04.csv --new-days-of-data 4 --recent-labeled-auc 0.79
python src/monitoring.py --recent-batch data/incoming/day_05_2026-08-05.csv   # negative control, should be clean

# 9. (Re)generate the architecture diagram.
python scripts/generate_diagram.py
```

Note on the ingestion ordering used to build this repo's fixtures: `train.py`
was run on the training table *before* `day_04`/`day_05` were ingested, so
`models/train_feature_stats.json` (the drift-monitoring reference) reflects
the pre-drift snapshot. Running step 2 straight through in one shot (as the
command above does) will fold all 5 days into training before you train —
that's fine for training, but if you want to reproduce the exact drift-catch
demo, ingest only `day_01`–`day_03` before running `train.py`, then ingest
`day_04`/`day_05` afterward. See the "Data pipeline" section of
`DESIGN_DOC.md` for why this ordering matters.



## Current results (from the committed artifacts)

- Production model: `candidate` (RandomForest) — see `models/registry.json`
  and `artifacts/eval/comparison_report.md` for the full promotion decision.
- Held-out test metrics: accuracy 0.786, ROC AUC 0.8224 (see `artifacts/eval/`).
- Online API latency: ~9.4ms avg / ~10.5ms p95 over 200 requests (`artifacts/eval/latency_report.json`).
- Batch scoring throughput: ~51,500 rows/sec for 7,043 rows (in-process, no I/O bottleneck).
- Drift check correctly flags `day_04` (injected drift) on 5 features and stays clean on `day_05` (negative control) — see `monitoring/`.

# Design Document: Customer Churn Prediction

## 1. Problem Definition and Metrics

**Use case.** Binary classification: given a telecom customer's account
attributes, predict whether they will churn (`Churn = Yes/No`). A churn
score feeds two downstream consumers: a retention team's daily outreach
list (batch) and a CRM widget an agent can query mid-call (online).

**Dataset.** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv),
7,043 customers, 21 raw columns (demographics, subscribed services,
contract/billing terms, tenure, charges, and the `Churn` label). Positive
rate (churn) is ~26.5% — moderately imbalanced, which drives the metric
choice below. One known data quirk: 11 customers with `tenure == 0` have a
blank `TotalCharges` string (they haven't been billed yet); this is handled
explicitly in `src/features.py` rather than silently coerced to `NaN`.

**Metrics.** ROC AUC is the primary model-selection metric because it is
threshold-independent and robust to the ~3:1 class imbalance — accuracy
alone can look good on a model that just leans toward predicting "No
churn." Precision, recall, and F1 are reported alongside AUC because they
map to the actual business trade-off: every predicted churner triggers a
retention offer with a real cost (recall too high → wasted incentives on
customers who weren't leaving; recall too low → preventable churn missed).
This project reports all four and lets a configurable promotion rule
(Section 3) act on AUC, but a real deployment would tune the decision
threshold against the retention team's cost-per-offer, not just use 0.5.

## 2. Data and Feature Design

**Six engineered features**, implemented once in `src/features.py::build_features`
and imported unchanged by training, the online API, and the batch scorer
(the training/serving-skew mitigation — see below):

1. `num_services` — count of subscribed add-ons + internet (aggregation).
2. `tenure_bucket` — new / established / loyal, a coarse lifecycle-stage encoding of tenure.
3. `avg_monthly_spend_ratio` — lifetime average spend (`TotalCharges / (tenure+1)`) vs. current `MonthlyCharges`; a ratio well below 1 flags a recent price increase or promo rolling off, a known churn precursor.
4. `contract_risk_score` — a domain-informed composite of contract length, payment method, and paperless billing, each independently correlated with churn.
5. `is_high_value_customer` — `MonthlyCharges` above the 75th percentile **of the training set**.
6. `has_internet_no_security` — interaction flag: has internet service but neither `OnlineSecurity` nor `TechSupport`.

**Online vs. offline.** In this dataset every raw input is available at
request time (it's a point-in-time account snapshot, not an event stream),
so features 1, 2, 4, and 6 are computed online, directly from the request
payload, with zero external lookups. Feature 5 is the interesting case: its
threshold is a statistic *of the training population*, not of any single
request, so it cannot be recomputed per-request without risking a
different cutoff at serving time than at training time (and a meaningless
cutoff for a single-row batch). It is fit once in `fit_feature_thresholds`
on the training split, persisted to `models/feature_thresholds.json`, and
loaded by `model_loader.py` alongside the model — a minimal "feature
store" in spirit: one small JSON file instead of a training/serving copy
of the same computation. In a system with true time-windowed aggregates
(e.g., "logins in the last 30 days"), this is exactly the pattern that
would generalize: an offline job writes aggregates to a feature table, and
serving does a lookup instead of a live recomputation.

**Training/serving skew mitigation.** The core guardrail here is
structural, not procedural: `build_features` is one function, imported by
`train.py`, `serving/app.py`, and `scripts/batch_score.py`. There is no
second implementation to drift out of sync. `tests/test_features.py`
includes a direct regression test for this: it featurizes one row as part
of a full batch (training path) and again as a lone single-row frame
(online-serving path) and asserts the two outputs are byte-for-byte
identical.

**Data pipeline step.** `src/data_ingestion.py` is the repeatable
micro-batch ingestion piece: it scans `data/incoming/` for CSV files not
yet recorded in `data/ingestion_log.jsonl`, validates each against the
required-column schema in `configs/config.yaml`, appends valid files to
`data/processed/training_data.csv`, and logs `{timestamp, source_file,
rows_ingested, cumulative_rows}` per file. It is idempotent (safe to
re-run on a schedule) and rejects files with missing columns outright
rather than silently dropping fields. `scripts/simulate_batches.py` is a
one-time fixture generator (not part of the repeatable pipeline) that
splits the raw dataset into a historical seed and 5 simulated daily
arrivals, with one day's `MonthlyCharges` and contract mix deliberately
shifted to give the monitoring step (Section 5) something real to catch.

## 3. Model Choice and Offline Evaluation

Two models share an identical preprocessing pipeline (`StandardScaler` on
numeric features, `OneHotEncoder` on categoricals) so the comparison
isolates model choice, not feature handling:

- **Baseline**: `LogisticRegression` — fast, interpretable, a reasonable
  floor for a tabular problem this size.
- **Candidate**: `RandomForestClassifier` (300 trees, max depth 8) — can
  capture non-linear interactions (e.g., contract type × tenure) the
  baseline cannot.

Split: 70% train / 15% val / 15% test, stratified on `Churn`, fit on the
pre-drift training snapshot (6,480 rows — see Section 4 on why `day_04`/
`day_05` were excluded from this snapshot). Validation results:

| Metric | Baseline (LogisticRegression) | Candidate (RandomForest) |
|---|---|---|
| Accuracy | 0.778 | 0.783 |
| ROC AUC | 0.839 | 0.836 |
| Precision | 0.604 | 0.614 |
| Recall | 0.502 | 0.517 |
| F1 | 0.548 | 0.561 |

**Promotion rule** (`configs/config.yaml: promotion`): promote the
candidate only if its AUC clears an absolute floor (0.75) *and* it does
not regress vs. the baseline by more than 0.01 AUC. Here the candidate's
AUC is 0.0032 below the baseline's — within the allowed regression margin
— and it clears the floor, and it wins on every other metric (precision,
recall, F1, accuracy), so `train.py` promotes it. The decision and full
reasoning are written to `artifacts/eval/comparison_report.md` and
`models/registry.json` every run, not just printed — this is what a
promotion gate in CI would read. On the held-out test set the promoted
model scores AUC 0.822, accuracy 0.786 — close enough to validation to
suggest limited overfitting from the RandomForest's depth constraint.

## 4. Serving and Inference Pattern

**Chosen pattern: hybrid** (online API + batch scorer), both reusing the
same `model_loader.py` (reads `models/registry.json` for "which model is
currently in production") and the same `features.py`.

- **Online** (`src/serving/app.py`, FastAPI): `POST /predict` takes raw
  customer fields and returns `{churn_probability, churn_prediction,
  model_version}`; `GET /health` reports liveness and the loaded model
  version. Justification (M2 framing): a CRM agent mid-call, or an
  automated retention workflow reacting to a support-ticket event, is a
  human/system *waiting* on the answer — this needs low, predictable
  latency, not high throughput on any single request.
- **Batch** (`scripts/batch_score.py`): scores an entire CSV in one
  process, no HTTP overhead. Justification: nightly scoring of the full
  customer base for a retention-outreach list has no one waiting on any
  individual row — throughput matters, not per-row latency.

**Measured performance** (`artifacts/eval/`): the online endpoint averaged
9.35ms latency (p50 9.19ms, p95 10.54ms, max 14.2ms) over 200 sequential
local requests, zero errors. Batch scoring processed all 7,043 rows in
0.137s (~51,500 rows/sec) — expected, since it pays feature engineering
and inference cost once per batch instead of once per HTTP round trip.
This asymmetry is the quantitative version of the pattern-choice argument
above: batch is roughly 500x more row-throughput-efficient, at the cost of
not being real-time.

**Containerization** (bonus): a minimal `Dockerfile` (`python:3.11-slim`,
copies `src/`, `models/`, `configs/`, runs `uvicorn`) is included but not
build-verified in this environment (no Docker daemon available here).

## 5. Data Pipeline, Monitoring, and Retraining

**Monitoring plan.**

- *Infra metrics*: request latency (avg/p95, measured above) and error
  rate — in production, a dashboard for the on-call engineer, alerting if
  p95 latency exceeds an SLO or error rate exceeds a small threshold.
- *Data/feature metrics*: null rates and out-of-range values (`tenure`,
  `MonthlyCharges`) on every incoming batch; mean/std drift on 6 numeric
  features and category-share drift on `Contract`, compared against a
  reference frozen at training time — a dashboard for the ML/data
  engineering team, alerting on any single flagged feature so a schema or
  upstream-pipeline change is caught before it reaches the model.
- *Model/business metrics*: AUC computed periodically against freshly
  labeled feedback (customers whose actual churn/no-churn outcome is now
  known) — a dashboard for the ML team and product stakeholders, alerting
  when it drops meaningfully below the last-validated baseline.

**Drift/quality check** (`src/monitoring.py`, all three implemented and
runnable, not just described): `check_schema` (required columns present),
`check_data_quality` (null rates, out-of-range values), and `check_drift`.
The drift check compares a recent batch against
`models/train_feature_stats.json` (the exact training-time reference,
frozen independent of how `training_data.csv` grows afterward) using a
**one-sample z-test** per numeric feature — `(recent_mean − ref_mean) /
(ref_std / √n_recent)` — rather than a flat percent-of-mean threshold. This
matters in practice: an early version of this check used a flat 10%
relative-mean-shift threshold and it false-positived on a clean 281-row
batch purely from sampling noise; the z-test correctly accounts for batch
size (a given mean shift is less surprising on a small sample) and made
the false positive disappear while still catching the real drift. Verified
against two real files: `day_04` (deliberately shifted `MonthlyCharges` +25%
and contract mix) correctly flags 4 numeric features (`MonthlyCharges`
z=6.72, `avg_monthly_spend_ratio` z=−19.56, `contract_risk_score` z=3.43,
`is_high_value_customer` z=7.19) and the `Contract` category (17.1pp share
shift); `day_05` (untouched) is checked as a negative control and correctly
reports no issues.

**Retraining trigger** (`should_retrain` in `src/monitoring.py`, a real
function, not pseudocode — the pseudocode form below is for readability):

```
retrain if ANY of:
    new_days_of_data        >= 3       # enough fresh data has accumulated
    baseline_auc - recent_labeled_auc  >= 0.03   # measured perf has degraded
    drift_score              >= 2      # 2+ features/categories flagged
```

Run against `day_04` with a simulated recent-feedback AUC of 0.79 (vs. the
production test AUC of 0.822), all three signals fire independently:
4 days of new data, a 0.032 AUC drop, and a drift score of 5. Run against
clean `day_05`, none fire. Any single signal is sufficient by design —
data volume, measured performance, and distributional drift each catch
failure modes the others miss (drift can precede a measurable AUC drop;
an AUC drop can happen with no detectable feature drift, e.g. a change in
customer behavior).

**Incident scenario.** Suppose the upstream billing system reworks its
export and silently drops the `Contract` column (a real "schema evolved
without telling anyone" failure). `check_schema` on the next ingested
batch immediately flags `missing_columns: ['Contract']` and the pipeline
skips the drift check for that batch rather than crashing deep inside
feature engineering or silently imputing a wrong default — this was
verified directly by dropping the column from a real batch file and
confirming `SCHEMA ALERT` fires and no drift/retrain computation proceeds
on bad data. Response: (1) freeze ingestion of new batches from that
source, (2) roll the production pointer in `models/registry.json` back to
the previously known-good model version if the current one is already
degrading (no retrain needed if the previous model still meets the
promotion bar), (3) fix the upstream export or add a backward-compatible
default, (4) backfill/re-ingest the affected batches once the schema is
restored, (5) retrain once enough corrected data has accumulated.

## 6. Trade-offs, Limitations, and Future Work

- **Static snapshot, not a real event stream.** Telco churn is a
  point-in-time dataset; "online vs. offline feature" is somewhat
  academic here since nothing requires a real feature store. The design
  (frozen reference stats, shared feature module, JSON-persisted
  thresholds) is deliberately the pattern that *would* generalize to a
  system with true time-windowed aggregates, but it is not stress-tested
  against one here.
- **Simple registry, no rollback tooling.** `models/registry.json` is a
  single JSON file with no history beyond "current baseline/candidate,"
  and no automated rollback command — Section 5's incident response
  describes rolling back manually. A real system would want versioned
  history and a one-command rollback.
- **Drift check is univariate.** Each feature is checked independently;
  it would miss a drift that only shows up in a *combination* of features
  (e.g., a multivariate distribution shift with unchanged marginals). A
  more mature version would add a multivariate check (e.g., population
  stability index on model score itself, or a classifier-based drift
  detector).
- **No real labeled feedback loop.** `recent_labeled_auc` in the
  retraining trigger is supplied manually via a CLI flag in this project;
  production would need an actual feedback pipeline joining predictions
  to eventual outcomes, with its own latency (churn labels only become
  known after the fact, often weeks later).
- **Threshold left at 0.5.** The API and evaluation both use the default
  0.5 cutoff; a production deployment should pick the operating threshold
  from the retention team's real cost-per-outreach vs. cost-per-missed-churn,
  which this project does not have.
- **Future work**: a real model registry (MLflow or similar) instead of
  one JSON file; a proper feature store for any future time-windowed
  features; canary/shadow deployment of the candidate before full
  promotion instead of an offline-only promotion gate; SHAP-based
  explanations surfaced alongside each prediction for the retention team.

# Model Comparison Report

Generated: 2026-08-07T08:19:18.791545+00:00

| Metric | Baseline (LogisticRegression) | Candidate (RandomForest) |
|---|---|---|
| accuracy | 0.7778 | 0.7829 |
| roc_auc | 0.8387 | 0.8355 |
| precision | 0.6037 | 0.6136 |
| recall | 0.5019 | 0.5172 |
| f1 | 0.5481 | 0.5613 |

**Promotion rule**: promote candidate only if AUC >= 0.75 and does not regress vs. baseline by more than 0.01.

**Decision**: Promote candidate

Reasons:
- candidate AUC 0.8355 >= floor 0.75
- baseline-candidate AUC gap 0.0032 <= allowed regression 0.01

**Production model**: `candidate` (version `candidate-20260807T081918`)

Held-out test metrics for production model: `{"accuracy": 0.786, "roc_auc": 0.8224, "precision": 0.6256, "recall": 0.5057, "f1": 0.5593, "n_samples": 972, "positive_rate": 0.2685}`

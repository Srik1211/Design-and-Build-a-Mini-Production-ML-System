"""Training pipeline: load -> split -> engineer features -> train baseline
and candidate models -> evaluate -> decide promotion -> save artifacts.

Trains two models on the SAME feature pipeline so the comparison isolates
the effect of model choice:
  - baseline:  LogisticRegression  (simple, fast, interpretable)
  - candidate: RandomForestClassifier (captures non-linear interactions)

Promotion rule (configs/config.yaml -> promotion): the candidate is only
promoted to production if its validation ROC AUC clears an absolute floor
AND it does not regress vs. the baseline by more than a small margin.
Otherwise the baseline stays in production. Either way both models and a
full comparison report are written to disk.

Usage: python src/train.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve
from src.features import build_features, fit_feature_thresholds, prepare_xy


def build_pipeline(cfg: dict, estimator) -> Pipeline:
    numeric_cols = cfg["features"]["numeric"]
    categorical_cols = cfg["features"]["categorical"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", estimator)])


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "n_samples": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
    }


def compute_reference_stats(train_featured: pd.DataFrame, cfg: dict) -> dict:
    """Reference distribution for drift monitoring, frozen at training
    time and independent of how the processed data table grows afterward.
    """
    numeric_cols = cfg["features"]["numeric"]
    categorical_cols = cfg["features"]["categorical"]
    numeric_stats = {
        col: {
            "mean": float(train_featured[col].mean()),
            "std": float(train_featured[col].std()),
        }
        for col in numeric_cols
    }
    categorical_stats = {
        col: train_featured[col].value_counts(normalize=True).round(4).to_dict()
        for col in categorical_cols
    }
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(train_featured)),
        "numeric": numeric_stats,
        "categorical": categorical_stats,
    }


def save_model(name: str, pipeline: Pipeline, metrics: dict, cfg: dict, n_train_rows: int) -> dict:
    model_dir = resolve(cfg["paths"]["models_dir"], name)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")

    version = f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    metadata = {
        "name": name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": n_train_rows,
        "feature_columns": {
            "numeric": cfg["features"]["numeric"],
            "categorical": cfg["features"]["categorical"],
        },
        "metrics_val": metrics,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    cfg = load_config()
    data_path = resolve(cfg["paths"]["processed_data"])
    if not data_path.exists():
        raise SystemExit(f"No training data at {data_path} - run src/data_ingestion.py first")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    target_col = cfg["target"]["column"]
    y_raw = df[target_col]

    train_df, temp_df = train_test_split(
        df,
        test_size=cfg["split"]["test_size"] + cfg["split"]["val_size"],
        stratify=y_raw,
        random_state=cfg["split"]["random_state"],
    )
    relative_test_size = cfg["split"]["test_size"] / (cfg["split"]["test_size"] + cfg["split"]["val_size"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[target_col],
        random_state=cfg["split"]["random_state"],
    )
    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Fit feature thresholds on TRAIN ONLY, then reuse for val/test/serving -
    # this is the piece that must never be refit at inference time.
    thresholds = fit_feature_thresholds(train_df, cfg["features"]["high_value_percentile"])

    X_train, y_train = prepare_xy(train_df, cfg, thresholds)
    X_val, y_val = prepare_xy(val_df, cfg, thresholds)
    X_test, y_test = prepare_xy(test_df, cfg, thresholds)

    models_dir = resolve(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "feature_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    train_featured = build_features(train_df, thresholds)
    reference_stats = compute_reference_stats(train_featured, cfg)
    with open(models_dir / "train_feature_stats.json", "w") as f:
        json.dump(reference_stats, f, indent=2)

    # --- Baseline ---
    baseline_pipeline = build_pipeline(
        cfg, LogisticRegression(max_iter=1000, random_state=cfg["training"]["random_state"])
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipeline, X_val, y_val)
    print("Baseline (LogisticRegression) val metrics:", baseline_metrics)

    # --- Candidate ---
    rf_cfg = cfg["training"]["random_forest"]
    candidate_pipeline = build_pipeline(
        cfg,
        RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            random_state=cfg["training"]["random_state"],
        ),
    )
    candidate_pipeline.fit(X_train, y_train)
    candidate_metrics = evaluate_model(candidate_pipeline, X_val, y_val)
    print("Candidate (RandomForest) val metrics:", candidate_metrics)

    baseline_meta = save_model("baseline", baseline_pipeline, baseline_metrics, cfg, len(train_df))
    candidate_meta = save_model("candidate", candidate_pipeline, candidate_metrics, cfg, len(train_df))

    with open(resolve(cfg["paths"]["eval_dir"], "baseline_eval.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    with open(resolve(cfg["paths"]["eval_dir"], "candidate_eval.json"), "w") as f:
        json.dump(candidate_metrics, f, indent=2)

    # --- Promotion decision ---
    promo_cfg = cfg["promotion"]
    auc_gap = baseline_metrics["roc_auc"] - candidate_metrics["roc_auc"]
    clears_floor = candidate_metrics["roc_auc"] >= promo_cfg["min_auc"]
    not_worse_than_baseline = auc_gap <= promo_cfg["max_auc_regression_vs_baseline"]
    promote_candidate = clears_floor and not_worse_than_baseline

    reasons = []
    reasons.append(
        f"candidate AUC {candidate_metrics['roc_auc']} {'>=' if clears_floor else '<'} "
        f"floor {promo_cfg['min_auc']}"
    )
    reasons.append(
        f"baseline-candidate AUC gap {round(auc_gap, 4)} "
        f"{'<=' if not_worse_than_baseline else '>'} allowed regression {promo_cfg['max_auc_regression_vs_baseline']}"
    )

    production_name = "candidate" if promote_candidate else "baseline"
    production_meta = candidate_meta if promote_candidate else baseline_meta

    # Final sanity check on the untouched test split for whichever model wins.
    production_pipeline = candidate_pipeline if promote_candidate else baseline_pipeline
    test_metrics = evaluate_model(production_pipeline, X_test, y_test)
    print(f"Production candidate ({production_name}) test metrics:", test_metrics)

    registry = {
        "production": production_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "baseline": {"path": "models/baseline/model.joblib", "version": baseline_meta["version"], "metrics_val": baseline_metrics},
            "candidate": {"path": "models/candidate/model.joblib", "version": candidate_meta["version"], "metrics_val": candidate_metrics},
        },
        "promotion_decision": {
            "promoted": "candidate" if promote_candidate else "none (kept baseline)",
            "reasons": reasons,
        },
        "production_test_metrics": test_metrics,
    }
    with open(resolve(cfg["paths"]["registry"]), "w") as f:
        json.dump(registry, f, indent=2)

    report_lines = [
        "# Model Comparison Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Metric | Baseline (LogisticRegression) | Candidate (RandomForest) |",
        "|---|---|---|",
    ]
    for metric in ["accuracy", "roc_auc", "precision", "recall", "f1"]:
        report_lines.append(f"| {metric} | {baseline_metrics[metric]} | {candidate_metrics[metric]} |")
    report_lines += [
        "",
        f"**Promotion rule**: promote candidate only if AUC >= {promo_cfg['min_auc']} "
        f"and does not regress vs. baseline by more than {promo_cfg['max_auc_regression_vs_baseline']}.",
        "",
        f"**Decision**: {'Promote candidate' if promote_candidate else 'Keep baseline in production'}",
        "",
        "Reasons:",
    ] + [f"- {r}" for r in reasons] + [
        "",
        f"**Production model**: `{production_name}` (version `{production_meta['version']}`)",
        "",
        f"Held-out test metrics for production model: `{json.dumps(test_metrics)}`",
    ]
    with open(resolve(cfg["paths"]["eval_dir"], "comparison_report.md"), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nDecision: {'PROMOTE candidate' if promote_candidate else 'KEEP baseline'} -> production = {production_name}")
    print(f"Artifacts written under {resolve(cfg['paths']['models_dir'])} and {resolve(cfg['paths']['eval_dir'])}")


if __name__ == "__main__":
    main()

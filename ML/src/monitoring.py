"""Data quality checks, drift detection, and the retraining trigger.

This is deliberately lightweight (no external monitoring stack) but each
check does real work against real files, not just pseudocode:

  1. check_schema        - are all required raw columns present? (catches
                            an upstream schema change immediately)
  2. check_data_quality   - null rates and out-of-range values on the raw
                            recent batch
  3. check_drift          - mean/std shift (numeric) and category-share
                            shift (categorical), recent batch vs. the
                            training-time reference frozen in
                            models/train_feature_stats.json
  4. should_retrain       - combines drift, a labeled-feedback AUC drop,
                            and new-data volume into a retrain/don't-retrain
                            decision

Usage:
    python src/monitoring.py --recent-batch data/incoming/day_04_2026-08-04.csv
    python src/monitoring.py --recent-batch data/incoming/day_05_2026-08-05.csv
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve
from src.features import build_features


def check_schema(df: pd.DataFrame, cfg: dict) -> dict:
    required = set(cfg["required_columns"])
    missing = sorted(required - set(df.columns))
    return {"ok": len(missing) == 0, "missing_columns": missing}


def check_data_quality(df: pd.DataFrame, cfg: dict) -> dict:
    mon_cfg = cfg["monitoring"]
    issues = []

    null_rates = df.isnull().mean().round(4)
    bad_null_cols = null_rates[null_rates > mon_cfg["max_null_rate"]]
    for col, rate in bad_null_cols.items():
        issues.append(f"column '{col}' has null rate {rate} > {mon_cfg['max_null_rate']}")

    if "tenure" in df.columns:
        lo, hi = mon_cfg["tenure_valid_range"]
        n_bad = int(((df["tenure"] < lo) | (df["tenure"] > hi)).sum())
        if n_bad:
            issues.append(f"{n_bad} rows have tenure outside [{lo}, {hi}]")

    if "MonthlyCharges" in df.columns:
        lo, hi = mon_cfg["monthly_charges_valid_range"]
        n_bad = int(((df["MonthlyCharges"] < lo) | (df["MonthlyCharges"] > hi)).sum())
        if n_bad:
            issues.append(f"{n_bad} rows have MonthlyCharges outside [{lo}, {hi}]")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "null_rates": null_rates.to_dict(),
        "n_rows": int(len(df)),
    }


def check_drift(reference_stats: dict, recent_featured: pd.DataFrame, cfg: dict) -> dict:
    mon_cfg = cfg["monitoring"]
    numeric_drift = {}
    categorical_drift = {}
    drift_score = 0

    for col, ref in reference_stats["numeric"].items():
        if col not in recent_featured.columns:
            continue
        n_recent = int(recent_featured[col].notna().sum())
        recent_mean = float(recent_featured[col].mean())
        ref_mean = ref["mean"]
        ref_std = ref["std"] or 1e-9

        # One-sample z-test: is the recent batch mean far from the training
        # mean relative to how much sampling noise we'd expect at this
        # batch size? Using pct-of-mean alone would flag small batches on
        # noise alone; the standard error term (ref_std / sqrt(n)) corrects
        # for that.
        standard_error = ref_std / math.sqrt(n_recent) if n_recent > 0 else float("inf")
        z_score = (recent_mean - ref_mean) / standard_error if standard_error > 0 else 0.0
        pct_change = abs(recent_mean - ref_mean) / (abs(ref_mean) if ref_mean != 0 else 1e-9)
        flagged = abs(z_score) > mon_cfg["numeric_drift_zscore_threshold"]
        numeric_drift[col] = {
            "reference_mean": round(ref_mean, 4),
            "recent_mean": round(recent_mean, 4),
            "pct_change": round(pct_change, 4),
            "z_score": round(z_score, 2),
            "n_recent": n_recent,
            "flagged": flagged,
        }
        if flagged:
            drift_score += 1

    for col, ref_shares in reference_stats["categorical"].items():
        if col not in recent_featured.columns:
            continue
        recent_shares = recent_featured[col].value_counts(normalize=True).round(4).to_dict()
        all_categories = set(ref_shares) | set(recent_shares)
        max_abs_diff = 0.0
        per_category = {}
        for cat in all_categories:
            ref_share = ref_shares.get(cat, 0.0)
            recent_share = recent_shares.get(cat, 0.0)
            diff = abs(recent_share - ref_share)
            per_category[cat] = {"reference": ref_share, "recent": round(recent_share, 4), "abs_diff": round(diff, 4)}
            max_abs_diff = max(max_abs_diff, diff)
        flagged = max_abs_diff > mon_cfg["categorical_drift_abs_threshold"]
        categorical_drift[col] = {"categories": per_category, "max_abs_diff": round(max_abs_diff, 4), "flagged": flagged}
        if flagged:
            drift_score += 1

    return {
        "numeric_drift": numeric_drift,
        "categorical_drift": categorical_drift,
        "drift_score": drift_score,
    }


def should_retrain(signals: dict, cfg: dict) -> dict:
    """Combine 3 independent retraining signals. Any one firing is enough
    to recommend a retrain; the pseudo-code version of this same logic is
    reproduced in DESIGN_DOC.md.

    signals expects:
        new_days_of_data: int
        drift_score: int                (from check_drift)
        baseline_auc: float | None
        recent_labeled_auc: float | None (AUC on freshly labeled feedback, if available)
    """
    trig_cfg = cfg["retraining_triggers"]
    reasons = []

    if signals.get("new_days_of_data", 0) >= trig_cfg["min_new_days_of_data"]:
        reasons.append(
            f"new_days_of_data={signals['new_days_of_data']} >= {trig_cfg['min_new_days_of_data']}"
        )

    baseline_auc = signals.get("baseline_auc")
    recent_auc = signals.get("recent_labeled_auc")
    if baseline_auc is not None and recent_auc is not None:
        auc_drop = baseline_auc - recent_auc
        if auc_drop >= trig_cfg["auc_drop_threshold"]:
            reasons.append(
                f"AUC dropped by {round(auc_drop, 4)} >= {trig_cfg['auc_drop_threshold']} "
                f"(baseline={baseline_auc}, recent={recent_auc})"
            )

    drift_score = signals.get("drift_score", 0)
    if drift_score >= trig_cfg["drift_score_threshold"]:
        reasons.append(f"drift_score={drift_score} >= {trig_cfg['drift_score_threshold']}")

    return {"retrain": len(reasons) > 0, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description="Run data quality + drift checks against a recent batch.")
    parser.add_argument("--recent-batch", type=str, required=True)
    parser.add_argument("--new-days-of-data", type=int, default=1, help="Signal for the retraining trigger demo")
    parser.add_argument("--recent-labeled-auc", type=float, default=None, help="Signal for the retraining trigger demo")
    args = parser.parse_args()

    cfg = load_config()
    models_dir = resolve(cfg["paths"]["models_dir"])

    with open(models_dir / "train_feature_stats.json") as f:
        reference_stats = json.load(f)
    with open(models_dir / "feature_thresholds.json") as f:
        thresholds = json.load(f)
    with open(resolve(cfg["paths"]["registry"])) as f:
        registry = json.load(f)

    recent_path = resolve(args.recent_batch)
    recent_df = pd.read_csv(recent_path)
    print(f"Checking {recent_path} ({len(recent_df)} rows) against reference "
          f"from {reference_stats['n_rows']} training rows...")

    schema_result = check_schema(recent_df, cfg)
    quality_result = check_data_quality(recent_df, cfg)

    if not schema_result["ok"]:
        print(f"SCHEMA ALERT: missing columns {schema_result['missing_columns']} - "
              f"skipping drift check (feature engineering would fail on this batch)")
        drift_result = {"numeric_drift": {}, "categorical_drift": {}, "drift_score": 0}
    else:
        recent_featured = build_features(recent_df, thresholds)
        drift_result = check_drift(reference_stats, recent_featured, cfg)

    for issue in quality_result["issues"]:
        print(f"QUALITY WARN: {issue}")
    for col, d in drift_result["numeric_drift"].items():
        if d["flagged"]:
            print(f"DRIFT WARN: numeric '{col}' z={d['z_score']} (shift {d['pct_change']*100:.1f}%, "
                  f"reference={d['reference_mean']}, recent={d['recent_mean']}, n={d['n_recent']})")
    for col, d in drift_result["categorical_drift"].items():
        if d["flagged"]:
            print(f"DRIFT WARN: categorical '{col}' max share shift {d['max_abs_diff']*100:.1f}pp")

    if schema_result["ok"] and quality_result["ok"] and drift_result["drift_score"] == 0:
        print("No data quality or drift issues detected.")

    retrain_signals = {
        "new_days_of_data": args.new_days_of_data,
        "drift_score": drift_result["drift_score"],
        "baseline_auc": registry["production_test_metrics"]["roc_auc"],
        "recent_labeled_auc": args.recent_labeled_auc,
    }
    retrain_decision = should_retrain(retrain_signals, cfg)
    print(f"\nRetraining decision: {'RETRAIN' if retrain_decision['retrain'] else 'no action'}")
    for reason in retrain_decision["reasons"]:
        print(f"  - {reason}")

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "recent_batch": str(recent_path),
        "schema": schema_result,
        "data_quality": quality_result,
        "drift": drift_result,
        "retrain_signals": retrain_signals,
        "retrain_decision": retrain_decision,
    }
    out_path = resolve(cfg["paths"]["monitoring_dir"], f"drift_report_{recent_path.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report -> {out_path}")


if __name__ == "__main__":
    main()

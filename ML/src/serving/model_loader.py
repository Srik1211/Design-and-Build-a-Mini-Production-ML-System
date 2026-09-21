"""Loads whichever model models/registry.json currently marks as
production, plus the feature thresholds it was trained with.

Used by both the online FastAPI app and the offline batch scorer so
"which model is live" and "how are features computed" are answered in
exactly one place.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_config, resolve


@dataclass
class ModelBundle:
    pipeline: object
    thresholds: dict
    production_name: str
    version: str
    metrics_val: dict


def load_production_model(cfg: dict = None) -> ModelBundle:
    cfg = cfg or load_config()

    registry_path = resolve(cfg["paths"]["registry"])
    if not registry_path.exists():
        raise FileNotFoundError(f"No model registry at {registry_path} - run src/train.py first")
    with open(registry_path) as f:
        registry = json.load(f)

    production_name = registry["production"]
    model_entry = registry["models"][production_name]

    thresholds_path = resolve(cfg["paths"]["models_dir"], "feature_thresholds.json")
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    pipeline = joblib.load(resolve(model_entry["path"]))

    return ModelBundle(
        pipeline=pipeline,
        thresholds=thresholds,
        production_name=production_name,
        version=model_entry["version"],
        metrics_val=model_entry["metrics_val"],
    )

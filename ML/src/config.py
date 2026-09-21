"""Single place that resolves the project root and loads configs/config.yaml.

Every other module gets paths/thresholds through here so training, serving,
ingestion and monitoring can never silently disagree about where data or
models live.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(*parts) -> Path:
    """Resolve a path relative to the ML/ project root."""
    return ROOT.joinpath(*parts)

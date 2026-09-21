"""Online inference service.

Exposes the production churn model as a request/response API. This is the
"a human (or a CRM system) is waiting for an answer" half of the serving
strategy described in DESIGN_DOC.md; scripts/batch_score.py is the
throughput-oriented offline half. Both reuse this same model bundle
loading code and the shared src/features.build_features, so there is no
separate "serving version" of the feature logic to drift out of sync.

Run from ML/:
    uvicorn src.serving.app:app --reload --port 8000
"""
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_config
from src.features import prepare_xy
from src.schema import CustomerRecord, HealthResponse, PredictionResponse
from src.serving.model_loader import load_production_model

app = FastAPI(title="Churn Prediction Service", version="1.0")

_cfg = load_config()
_bundle = load_production_model(_cfg)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version=_bundle.version,
        production_model=_bundle.production_name,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord):
    row = pd.DataFrame([record.model_dump()])
    try:
        X, _ = prepare_xy(row, _cfg, _bundle.thresholds)
    except Exception as exc:  # malformed/unexpected category values, etc.
        raise HTTPException(status_code=422, detail=f"Feature construction failed: {exc}")

    proba = float(_bundle.pipeline.predict_proba(X)[0, 1])
    prediction = _cfg["target"]["positive_label"] if proba >= 0.5 else "No"

    return PredictionResponse(
        customerID=record.customerID,
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        model_version=_bundle.version,
    )

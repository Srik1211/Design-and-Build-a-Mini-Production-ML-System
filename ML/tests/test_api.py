from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "customerID": "0000-TEST",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": "427.5",
}


def test_health_returns_ok_and_model_version():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert body["production_model"] in {"baseline", "candidate"}


def test_predict_happy_path():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["customerID"] == "0000-TEST"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
    assert body["model_version"]


def test_predict_high_risk_profile_scores_high():
    # month-to-month + electronic check + no security add-ons is a
    # textbook high-churn-risk profile for this dataset.
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.json()["churn_probability"] > 0.5


def test_predict_rejects_missing_required_field():
    incomplete = dict(VALID_PAYLOAD)
    del incomplete["MonthlyCharges"]
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_tenure():
    bad = dict(VALID_PAYLOAD, tenure=999)
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422

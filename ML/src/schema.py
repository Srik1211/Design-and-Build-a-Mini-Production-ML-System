"""Pydantic request/response models for the serving API.

Field names deliberately mirror the raw Telco columns so a request body
can be turned into a one-row DataFrame and passed straight into
`features.build_features` -- no separate mapping layer to keep in sync.
"""
from typing import Optional, Union

from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    customerID: Optional[str] = Field(default=None, description="Optional identifier, not used as a model input")
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Union[float, str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "0000-DEMO",
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
        }
    }


class PredictionResponse(BaseModel):
    customerID: Optional[str] = None
    churn_probability: float
    churn_prediction: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    production_model: str

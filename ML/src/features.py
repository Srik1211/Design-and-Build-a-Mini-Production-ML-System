"""Shared feature engineering.

This module is imported, unchanged, by src/train.py, src/serving/app.py and
scripts/batch_score.py. That is the training/serving-skew mitigation for
this project: there is exactly one implementation of every feature, so a
single customer record produces the same feature vector whether it runs
through the training pipeline or the live API.

The one feature that needs a value learned from the training set
(`is_high_value_customer`'s MonthlyCharges cutoff) is NOT recomputed at
serving time -- it is fit once in `fit_feature_thresholds`, persisted to
disk alongside the model, and passed back into `build_features` at
inference time. Recomputing a percentile per-request (or over whatever
batch happens to be in memory) is a classic source of skew; this avoids it.
"""
from typing import Optional

import numpy as np
import pandas as pd

# Columns treated as "add-on services" for the num_services aggregation.
SERVICE_COLUMNS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CONTRACT_WEIGHTS = {"Month-to-month": 3, "One year": 1, "Two year": 0}
PAYMENT_RISK_METHODS = {"Electronic check"}


def _clean_total_charges(df: pd.DataFrame) -> pd.Series:
    """TotalCharges arrives as a string in the raw data and is blank for
    the ~11 customers with tenure == 0 (brand new, not yet billed). Coerce
    to numeric and treat those blanks as 0 rather than dropping rows.
    """
    charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return charges.fillna(0.0)


def fit_feature_thresholds(df: pd.DataFrame, high_value_percentile: float) -> dict:
    """Compute the training-set-derived constants that build_features needs.
    Call this ONCE on the training split and persist the result; never
    recompute it at serving time.
    """
    return {
        "high_value_cutoff": float(df["MonthlyCharges"].quantile(high_value_percentile)),
    }


def build_features(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add engineered columns to a raw customer dataframe (or single-row
    frame, for online serving) and return the augmented frame. Does not
    mutate the input.

    `thresholds` must come from `fit_feature_thresholds` on the training
    data (see module docstring).
    """
    out = df.copy()

    out["TotalCharges"] = _clean_total_charges(out)

    # 1. num_services - aggregation over subscribed add-ons + internet.
    has_service = pd.DataFrame(
        {col: (out[col] == "Yes").astype(int) for col in SERVICE_COLUMNS}
    )
    has_internet = (out["InternetService"] != "No").astype(int)
    out["num_services"] = has_service.sum(axis=1) + has_internet

    # 2. tenure_bucket - coarse time-window encoding of customer lifecycle stage.
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 12, 36, np.inf],
        labels=["new", "established", "loyal"],
    ).astype(str)

    # 3. avg_monthly_spend_ratio - actual lifetime average spend vs current
    # sticker MonthlyCharges. A ratio well below 1 means the customer is
    # paying much more now than their history would suggest (e.g. a promo
    # rolled off), a known churn precursor.
    avg_monthly_spend = out["TotalCharges"] / (out["tenure"] + 1)
    out["avg_monthly_spend_ratio"] = avg_monthly_spend / out["MonthlyCharges"].replace(0, np.nan)
    out["avg_monthly_spend_ratio"] = out["avg_monthly_spend_ratio"].fillna(1.0)

    # 4. contract_risk_score - domain-informed composite of contract length,
    # payment method and billing type, each independently correlated with
    # churn in the telco literature.
    contract_component = out["Contract"].map(CONTRACT_WEIGHTS).fillna(3)
    payment_component = out["PaymentMethod"].isin(PAYMENT_RISK_METHODS).astype(int) * 2
    paperless_component = (out["PaperlessBilling"] == "Yes").astype(int)
    out["contract_risk_score"] = contract_component + payment_component + paperless_component

    # 5. is_high_value_customer - threshold fit on training data, reused as-is.
    out["is_high_value_customer"] = (
        out["MonthlyCharges"] >= thresholds["high_value_cutoff"]
    ).astype(int)

    # 6. has_internet_no_security - interaction flag: internet service with
    # neither OnlineSecurity nor TechSupport, a higher-risk combination than
    # either signal alone.
    out["has_internet_no_security"] = (
        (out["InternetService"] != "No")
        & (out["OnlineSecurity"] == "No")
        & (out["TechSupport"] == "No")
    ).astype(int)

    return out


def prepare_xy(
    df: pd.DataFrame,
    cfg: dict,
    thresholds: dict,
    target_column: Optional[str] = None,
    positive_label: Optional[str] = None,
):
    """Build features and split into the model-ready X (selected columns
    only) and y (binary target), applying the same feature list train and
    serve both rely on via cfg['features'].
    """
    featured = build_features(df, thresholds)
    numeric_cols = cfg["features"]["numeric"]
    categorical_cols = cfg["features"]["categorical"]
    X = featured[numeric_cols + categorical_cols]

    y = None
    target_column = target_column or cfg["target"]["column"]
    positive_label = positive_label or cfg["target"]["positive_label"]
    if target_column in featured.columns:
        y = (featured[target_column] == positive_label).astype(int)

    return X, y

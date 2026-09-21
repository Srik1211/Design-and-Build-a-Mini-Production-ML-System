import pandas as pd

from src.config import load_config, resolve
from src.features import build_features, fit_feature_thresholds, prepare_xy

CFG = load_config()


def _sample_df():
    return pd.read_csv(resolve("data/raw/historical_seed.csv")).head(50)


def test_build_features_adds_expected_columns():
    df = _sample_df()
    thresholds = fit_feature_thresholds(df, CFG["features"]["high_value_percentile"])
    featured = build_features(df, thresholds)

    expected_new_cols = {
        "num_services",
        "tenure_bucket",
        "avg_monthly_spend_ratio",
        "contract_risk_score",
        "is_high_value_customer",
        "has_internet_no_security",
    }
    assert expected_new_cols.issubset(set(featured.columns))


def test_build_features_no_unexpected_nans():
    df = _sample_df()
    thresholds = fit_feature_thresholds(df, CFG["features"]["high_value_percentile"])
    featured = build_features(df, thresholds)

    engineered_cols = [
        "num_services", "tenure_bucket", "avg_monthly_spend_ratio",
        "contract_risk_score", "is_high_value_customer", "has_internet_no_security",
    ]
    assert not featured[engineered_cols].isnull().any().any()


def test_blank_total_charges_does_not_crash():
    df = _sample_df().copy()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "  # exact quirk present in the raw Telco data

    thresholds = fit_feature_thresholds(df, CFG["features"]["high_value_percentile"])
    featured = build_features(df, thresholds)

    assert featured.loc[0, "TotalCharges"] == 0.0
    assert pd.notna(featured.loc[0, "avg_monthly_spend_ratio"])


def test_single_row_matches_batch_row():
    """The core training/serving-skew guardrail: featurizing one row alone
    (as the online API does) must produce identical values to featurizing
    it as part of a larger batch (as training does).
    """
    df = _sample_df()
    thresholds = fit_feature_thresholds(df, CFG["features"]["high_value_percentile"])

    X_batch, _ = prepare_xy(df, CFG, thresholds)
    single_row = df.iloc[[3]]
    X_single, _ = prepare_xy(single_row, CFG, thresholds)

    expected = X_batch.iloc[[3]].reset_index(drop=True)
    actual = X_single.reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_prepare_xy_returns_binary_target_when_present():
    df = _sample_df()
    thresholds = fit_feature_thresholds(df, CFG["features"]["high_value_percentile"])
    _, y = prepare_xy(df, CFG, thresholds)

    assert y is not None
    assert set(y.unique()).issubset({0, 1})

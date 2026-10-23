"""Feature engineering for the U.S. recession forecasting project."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "macroeconomic_monthly.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "recession_features_6m.csv"
)


def load_monthly_data(input_path=DEFAULT_INPUT_PATH):
    """Load the cleaned monthly macroeconomic dataset."""
    return pd.read_csv(
        input_path,
        index_col=0,
        parse_dates=True,
    )


def create_target(df, forecast_horizon=6):
    """Create the recession target for exactly `forecast_horizon` months ahead."""
    df = df.copy()
    df[f"recession_{forecast_horizon}m"] = df["recession"].shift(
        -forecast_horizon
    )
    return df


def create_features(df, forecast_horizon=6):
    """Create the predictor features and future recession target."""
    df = create_target(df, forecast_horizon=forecast_horizon)

    features = pd.DataFrame(index=df.index)

    # Current levels
    level_features = [
        "unemployment_rate",
        "fed_funds_rate",
        "yield_spread_10y_2y",
        "yield_spread_10y_3m",
        "consumer_sentiment",
        "credit_spread",
    ]

    for column in level_features:
        features[column] = df[column]

    # Labor market
    features["unemployment_change_3m"] = df["unemployment_rate"].diff(3)
    features["unemployment_change_6m"] = df["unemployment_rate"].diff(6)
    features["unemployment_change_12m"] = df["unemployment_rate"].diff(12)

    features["initial_claims"] = df["initial_claims"]
    features["initial_claims_change_3m"] = (
        df["initial_claims"].pct_change(3) * 100
    )
    features["initial_claims_change_6m"] = (
        df["initial_claims"].pct_change(6) * 100
    )

    # Inflation
    features["inflation_yoy"] = df["cpi"].pct_change(12) * 100

    # Economic activity
    features["industrial_production_growth_3m"] = (
        df["industrial_production"].pct_change(3) * 100
    )
    features["industrial_production_growth_12m"] = (
        df["industrial_production"].pct_change(12) * 100
    )

    # Housing
    features["housing_starts_change_3m"] = (
        df["housing_starts"].pct_change(3) * 100
    )
    features["housing_starts_change_12m"] = (
        df["housing_starts"].pct_change(12) * 100
    )

    # Consumer sentiment
    features["consumer_sentiment_change_3m"] = (
        df["consumer_sentiment"].diff(3)
    )
    features["consumer_sentiment_change_6m"] = (
        df["consumer_sentiment"].diff(6)
    )

    # Yield-curve inversion indicators
    features["yield_curve_inverted_10y_2y"] = (
        df["yield_spread_10y_2y"] < 0
        ).astype(int)
    features["yield_curve_inverted_10y_3m"] = (
        df["yield_spread_10y_3m"] < 0
    ).astype(int)

    # Credit conditions
    features["credit_spread_change_3m"] = df["credit_spread"].diff(3)
    features["credit_spread_change_6m"] = df["credit_spread"].diff(6)

    # Target: recession status exactly N months later
    features["target"] = df[f"recession_{forecast_horizon}m"]

    return features


def clean_features(features):
    """Remove rows made incomplete by lags, percentage changes, or target shifting."""
    return features.dropna().copy()


def build_feature_dataset(
    input_path=DEFAULT_INPUT_PATH,
    forecast_horizon=6,
):
    """Load monthly data and run the complete feature-engineering pipeline."""
    df = load_monthly_data(input_path)
    features = create_features(df, forecast_horizon=forecast_horizon)
    return clean_features(features)


def save_feature_dataset(
    df,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """Save the engineered feature dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


if __name__ == "__main__":
    features = build_feature_dataset(forecast_horizon=6)
    save_feature_dataset(features)

    print("Feature dataset saved")
    print(f"Observations: {len(features)}")
    print(f"Features: {features.shape[1] - 1}")
    print(
        f"Period: {features.index.min().date()} "
        f"to {features.index.max().date()}"
    )

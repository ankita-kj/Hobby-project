"""Utilities for downloading and preprocessing macroeconomic FRED data."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

SERIES = {
    # Labor market
    "unemployment_rate": "UNRATE",
    "initial_claims": "ICSA",

    # Inflation
    "cpi": "CPIAUCSL",

    # Interest rates
    "fed_funds_rate": "FEDFUNDS",

    # Yield curve
    "yield_spread_10y_2y": "T10Y2Y",
    "yield_spread_10y_3m": "T10Y3M",

    # Economic activity
    "industrial_production": "INDPRO",

    # Housing
    "housing_starts": "HOUST",

    # Consumer sentiment
    "consumer_sentiment": "UMCSENT",

    # Financial markets
    "sp500": "SP500",

    # Credit conditions
    "credit_spread": "BAA10Y",

    # Recession indicator
    "recession": "USREC",
}

def get_fred_client(env_path=ENV_PATH):
    """Create an authenticated FRED client using FRED_API_KEY from .env."""
    load_dotenv(env_path)

    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError(
            "FRED_API_KEY was not found. Add it to your .env file."
        )

    return Fred(api_key=api_key)

def download_fred_data(
    fred,
    series=SERIES,
    observation_start="1960-01-01",
):
    """Download all configured FRED series and return them as a dictionary."""
    data = {}

    for name, series_id in series.items():
        data[name] = fred.get_series(
            series_id,
            observation_start=observation_start,
        )

    return data

def convert_to_monthly(data, exclude=("sp500",)):
    """Convert downloaded series to monthly averages.

    SP500 is excluded in the current project because the available FRED
    history begins much later than the other indicators.
    """
    monthly_data = {}

    for name, values in data.items():
        if name in exclude:
            continue

        monthly_data[name] = values.resample("MS").mean()

    return pd.DataFrame(monthly_data)

def clean_monthly_data(df, start_date="1986-01-01"):
    """Restrict the dataset to the common analysis period and remove missing rows."""
    return df.loc[start_date:].dropna().copy()

def build_dataset(
    env_path=ENV_PATH,
    observation_start="1960-01-01",
    analysis_start="1986-01-01",
):
    """Run the complete data collection and preprocessing pipeline."""
    fred = get_fred_client(env_path)

    raw_data = download_fred_data(
        fred,
        observation_start=observation_start,
    )

    monthly_data = convert_to_monthly(raw_data)
    clean_data = clean_monthly_data(
        monthly_data,
        start_date=analysis_start,
    )

    return clean_data

def save_dataset(
    df,
    output_path=PROJECT_ROOT / "data" / "processed" / "macroeconomic_monthly.csv",
):
    """Save the processed monthly dataset to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

if __name__ == "__main__":
    df = build_dataset()
    save_dataset(df)

    print("Dataset saved")
    print(f"Observations: {len(df)}")
    print(
        f"Period: {df.index.min().date()} "
        f"to {df.index.max().date()}"
    )
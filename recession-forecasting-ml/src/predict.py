"""Generate a current six-month-ahead U.S. recession forecast.

This module downloads the latest FRED data, recreates the project's feature
engineering, trains the three forecasting models on observations whose
six-month-ahead recession target is known, and predicts from the newest
complete predictor row.

The output is intended for research/educational use, not as an official
economic forecast or financial advice.
"""

from pathlib import Path
import sys

import pandas as pd

# Support both:
#   python src/predict.py
# and:
#   from src.predict import generate_forecast
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataset
from src.features import create_features
from src.models import (
    build_logistic_model,
    build_random_forest_model,
    build_xgboost_model,
)

FORECAST_HORIZON = 6


def prepare_training_and_latest(monthly_data, forecast_horizon=FORECAST_HORIZON):
    """Create labeled training data and the newest complete predictor row.

    `create_features` intentionally leaves the newest target values missing,
    because recession status six months in the future is not yet known.
    Those rows are still valid for live prediction as long as every predictor
    is available.
    """
    feature_data = create_features(
        monthly_data,
        forecast_horizon=forecast_horizon,
    )

    predictor_columns = [
        column for column in feature_data.columns
        if column != "target"
    ]

    # Historical observations for which the future target is known.
    training_data = feature_data.dropna(
        subset=predictor_columns + ["target"]
    ).copy()

    # Recent observations may have an unknown future target but can still be
    # used for forecasting if all predictors are available.
    prediction_candidates = feature_data.dropna(
        subset=predictor_columns
    ).copy()

    if training_data.empty:
        raise ValueError("No complete labeled observations are available for training.")

    if prediction_candidates.empty:
        raise ValueError("No complete predictor rows are available for forecasting.")

    latest_features = prediction_candidates[predictor_columns].iloc[[-1]].copy()

    X_train = training_data[predictor_columns]
    y_train = training_data["target"].astype(int)

    # Safety check: never train on a row whose predictor month is at or after
    # the month being forecast from.
    latest_date = latest_features.index[0]
    mask = X_train.index < latest_date
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]

    return X_train, y_train, latest_features


def fit_forecast_models(X_train, y_train):
    """Fit the three models used in the project."""
    models = {
        "Logistic Regression": build_logistic_model(),
        "Random Forest": build_random_forest_model(),
        "XGBoost": build_xgboost_model(y_train),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def predict_probabilities(models, latest_features):
    """Return six-month-ahead recession probabilities for each model."""
    return {
        name: float(model.predict_proba(latest_features)[0, 1])
        for name, model in models.items()
    }


def generate_forecast(forecast_horizon=FORECAST_HORIZON):
    """Run the full live forecasting pipeline.

    Returns a dictionary designed to be consumed directly by the future
    Streamlit dashboard.
    """
    monthly_data = build_dataset()

    X_train, y_train, latest_features = prepare_training_and_latest(
        monthly_data,
        forecast_horizon=forecast_horizon,
    )

    models = fit_forecast_models(X_train, y_train)
    probabilities = predict_probabilities(models, latest_features)

    latest_date = pd.Timestamp(latest_features.index[0])
    target_date = latest_date + pd.DateOffset(months=forecast_horizon)

    results = pd.DataFrame(
        {
            "Model": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }
    )

    return {
        "latest_data_month": latest_date,
        "target_month": target_date,
        "forecast_horizon": forecast_horizon,
        "probabilities": probabilities,
        "results": results,
        "latest_features": latest_features,
        "training_observations": len(X_train),
        "training_start": X_train.index.min(),
        "training_end": X_train.index.max(),
        "models": models,
    }


def print_forecast(forecast):
    """Print a readable terminal summary."""
    print("\nU.S. Recession Forecast — Research Model")
    print("=" * 43)
    print(
        "Latest complete data month:",
        forecast["latest_data_month"].strftime("%B %Y"),
    )
    print(
        "Forecast target month:",
        forecast["target_month"].strftime("%B %Y"),
    )
    print(
        f"Horizon: {forecast['forecast_horizon']} months\n"
    )

    for model, probability in forecast["probabilities"].items():
        print(f"{model:<20} {probability:>7.1%}")

    print(
        "\nNote: These are model-estimated probabilities based on historical "
        "relationships in revised FRED data. They are not an official "
        "recession forecast or financial advice."
    )


if __name__ == "__main__":
    forecast = generate_forecast()
    print_forecast(forecast)

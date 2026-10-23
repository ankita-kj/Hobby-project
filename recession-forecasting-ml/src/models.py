"""Model definitions for the U.S. recession forecasting project."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_baseline_model():
    """Return the prior-probability dummy classifier used as a baseline."""
    return DummyClassifier(strategy="prior")


def build_logistic_model():
    """Return the class-balanced logistic regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])


def build_random_forest_model():
    """Return the class-balanced Random Forest model."""
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def calculate_scale_pos_weight(y_train):
    """Calculate the negative-to-positive class ratio for XGBoost."""
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError(
            "XGBoost scale_pos_weight cannot be calculated because "
            "the training data contains no positive observations."
        )

    return negative_count / positive_count


def build_xgboost_model(y_train):
    """Return the XGBoost model using the training-set class imbalance."""
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )


def build_models(y_train):
    """Return all models used in the project."""
    return {
        "Baseline": build_baseline_model(),
        "Logistic Regression": build_logistic_model(),
        "Random Forest": build_random_forest_model(),
        "XGBoost": build_xgboost_model(y_train),
    }

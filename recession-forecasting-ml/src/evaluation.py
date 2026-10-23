"""Evaluation utilities for the U.S. recession forecasting project."""

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a fitted classifier on the holdout test set."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
    }

    return results, predictions, probabilities


def time_series_cv(
    model_name,
    model_builder,
    X,
    y,
    n_splits=5,
):
    """Evaluate a model with expanding-window TimeSeriesSplit.

    Folds that contain only one target class in either training or
    validation are skipped because all evaluation metrics would not
    be meaningful.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        if (
            y_fold_train.nunique() < 2
            or y_fold_val.nunique() < 2
        ):
            print(
                f"Skipping fold {fold}: "
                "only one class present."
            )
            continue

        model = model_builder(y_fold_train)
        model.fit(X_fold_train, y_fold_train)

        predictions = model.predict(X_fold_val)
        probabilities = model.predict_proba(
            X_fold_val
        )[:, 1]

        fold_results.append({
            "Model": model_name,
            "Fold": fold,
            "Precision": precision_score(
                y_fold_val,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_fold_val,
                predictions,
                zero_division=0,
                  ),
            "F1": f1_score(
                y_fold_val,
                predictions,
                zero_division=0,
            ),
            "ROC-AUC": roc_auc_score(
                y_fold_val,
                probabilities,
            ),
            "PR-AUC": average_precision_score(
                y_fold_val,
                probabilities,
            ),
        })

    return pd.DataFrame(fold_results)


def summarize_cv(cv_results):
    """Return mean and standard deviation of CV metrics by model."""
    metrics = [
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    return (
        cv_results
        .groupby("Model")[metrics]
    )

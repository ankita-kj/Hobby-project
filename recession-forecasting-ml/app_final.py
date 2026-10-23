"""Streamlit dashboard for the U.S. recession forecasting project."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import generate_forecast


st.set_page_config(
    page_title="U.S. Recession Forecast",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        .eyebrow {
            color: #8b93a5;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .hero-title {
            font-size: 2.7rem;
            font-weight: 750;
            letter-spacing: -0.04em;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            color: #8b93a5;
            font-size: 1.05rem;
            margin-bottom: 1.8rem;
        }
        .forecast-card {
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            min-height: 150px;
        }
        .model-name {
            color: #8b93a5;
            font-size: 0.82rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .probability {
            font-size: 2.4rem;
            font-weight: 750;
            margin: 0.25rem 0;
        }
        .small-note {
            color: #8b93a5;
            font-size: 0.88rem;
        }
        .section-space {
            margin-top: 2.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast():
    return generate_forecast()


@st.cache_data(show_spinner=False)
def load_historical_probabilities():
    """Recreate the chronological holdout probability paths used in the project."""
    path = PROJECT_ROOT / "data" / "processed" / "recession_features_6m.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    X = df.drop(columns="target")
    y = df["target"].astype(int)

    split_index = int(len(df) * 0.80)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    from src.models import (
        build_logistic_model,
        build_random_forest_model,
        build_xgboost_model,
    )

    logistic = build_logistic_model()
    forest = build_random_forest_model()
    xgb = build_xgboost_model(y_train)

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    probabilities = pd.DataFrame(
        {
            "Logistic Regression": logistic.predict_proba(X_test)[:, 1],
            "Random Forest": forest.predict_proba(X_test)[:, 1],
            "XGBoost": xgb.predict_proba(X_test)[:, 1],
        },
        index=X_test.index,
    )

    positive_months = y_test[y_test == 1].index
    return probabilities, positive_months


def probability_card(model_name, probability):
    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="model-name">{model_name}</div>
            <div class="probability">{probability:.1%}</div>
            <div class="small-note">6-month-ahead estimate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Header
st.markdown('<div class="eyebrow">Macroeconomic ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">U.S. Recession Forecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Six-month-ahead recession estimates using U.S. macroeconomic data from FRED.</div>',
    unsafe_allow_html=True,
)

refresh_col, _ = st.columns([1, 5])
with refresh_col:
    if st.button("Refresh latest data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("Downloading the latest economic data and running the models..."):
        forecast = load_forecast()
except Exception as exc:
    st.error("The forecast could not be generated.")
    st.exception(exc)
    st.stop()

latest_month = forecast["latest_data_month"]
target_month = forecast["target_month"]
probabilities = forecast["probabilities"]
latest_features = forecast["latest_features"].iloc[0]

# Forecast metadata
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
meta1, meta2, meta3 = st.columns(3)
meta1.metric("Latest complete data", latest_month.strftime("%B %Y"))
meta2.metric("Forecast target", target_month.strftime("%B %Y"))
meta3.metric("Forecast horizon", f'{forecast["forecast_horizon"]} months')

st.divider()

# Model cards
st.subheader("Model forecasts")
st.caption(
    "Each model is shown separately because the project has not established "
    "a validated ensemble weighting."
)

c1, c2, c3 = st.columns(3)
with c1:
    probability_card("Logistic Regression", probabilities["Logistic Regression"])
with c2:
    probability_card("Random Forest", probabilities["Random Forest"])
with c3:
    probability_card("XGBoost", probabilities["XGBoost"])

# Forecast comparison
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("Forecast comparison")

comparison = (
    forecast["results"]
    .set_index("Model")[["Probability"]]
    .sort_values("Probability", ascending=False)
    * 100
)
comparison.columns = ["Probability (%)"]
st.bar_chart(comparison, y="Probability (%)", horizontal=True)

# Historical probability paths
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("Historical model probabilities")
st.caption(
    "Predicted six-month-ahead recession probabilities during the chronological "
    "holdout period. The historical test set contains very few recession target months."
)

try:
    historical_probabilities, positive_months = load_historical_probabilities()
    st.line_chart(historical_probabilities)

    if len(positive_months) > 0:
        formatted_months = ", ".join(
            pd.Timestamp(date).strftime("%b %Y") for date in positive_months
        )
        st.caption(f"Positive recession target predictor months in the holdout set: {formatted_months}.")
except Exception as exc:
    st.info("Historical probability chart is unavailable.")
    st.caption(str(exc))

# Latest inputs
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("Latest model inputs")
st.caption(
    f"Selected feature values used by the models for the {latest_month.strftime('%B %Y')} forecast."
)

signal_names = {
    "unemployment_rate": ("Unemployment rate", "%"),
    "unemployment_change_12m": ("12-month unemployment change", " pp"),
    "inflation_yoy": ("Inflation (YoY)", "%"),
    "yield_spread_10y_2y": ("10Y–2Y yield spread", " pp"),
    "housing_starts_change_12m": ("12-month housing starts change", "%"),
    "consumer_sentiment": ("Consumer sentiment", ""),
}

signal_cols = st.columns(3)
for i, (feature, (label, suffix)) in enumerate(signal_names.items()):
    value = latest_features.get(feature)
    if pd.notna(value):
        signal_cols[i % 3].metric(label, f"{value:.2f}{suffix}")

# Feature importance
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("What drives the models?")
st.caption(
    "Cross-model importance ranking from the project analysis. Lower average rank means "
    "the feature was more prominent across Logistic Regression, Random Forest, and XGBoost."
)

try:
    models = forecast["models"]
    feature_names = forecast["latest_features"].columns

    logistic_importance = pd.Series(
        np.abs(models["Logistic Regression"].named_steps["model"].coef_[0]),
        index=feature_names,
    )
    rf_importance = pd.Series(
        models["Random Forest"].feature_importances_,
        index=feature_names,
    )
    xgb_importance = pd.Series(
        models["XGBoost"].feature_importances_,
        index=feature_names,
    )

    importance = pd.DataFrame(
        {
            "Logistic Regression": logistic_importance,
            "Random Forest": rf_importance,
            "XGBoost": xgb_importance,
        }
    )
    importance["Average rank"] = (
        importance.rank(ascending=False).mean(axis=1)
    )

    top_importance = (
        importance.sort_values("Average rank")
        .head(10)[["Average rank"]]
        .sort_values("Average rank", ascending=False)
    )
    st.bar_chart(top_importance, horizontal=True)
    st.caption(
        "Importance describes model usage, not economic causality. Values from different "
        "model families are compared by rank rather than raw magnitude."
    )
except Exception as exc:
    st.info("Feature-importance chart is unavailable.")
    st.caption(str(exc))

# About
st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)
st.subheader("About the model")

about1, about2, about3 = st.columns(3)
about1.metric("Predictor features", latest_features.shape[0])
about2.metric("Training observations", forecast["training_observations"])
about3.metric(
    "Training period",
    f'{forecast["training_start"].year}–{forecast["training_end"].year}',
)

st.markdown(
    """
    The models use labor-market, inflation, interest-rate, yield-curve,
    industrial-production, housing, consumer-sentiment, and credit-market
    indicators. Logistic Regression, Random Forest, and XGBoost are trained
    on the same engineered feature set.

    The target is whether the U.S. economy is classified as being in recession
    **exactly six months after the predictor month**.
    """
)

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        - Monthly macroeconomic series are downloaded from FRED.
        - The feature pipeline creates changes, growth rates, yield-curve inversion
          indicators, and other derived predictors.
        - The historical dataset contains relatively few recession episodes, creating
          substantial class imbalance.
        - Historical evaluation showed that recession detection varies considerably
          across economic episodes.
        - FRED data used here are current/revised observations rather than real-time
          historical vintages.
        - Publication lags and later data revisions are not fully modeled.
        - The three model probabilities should be interpreted separately; no validated
          ensemble probability is reported.
        """
    )

st.divider()
st.caption(
    "Research and educational project only. Model-estimated probabilities are "
    "not an official recession forecast, investment recommendation, or financial advice."
)

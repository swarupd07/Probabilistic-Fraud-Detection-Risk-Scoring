# app.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils import load_model, load_data, get_feature_columns
from decision_engine import evaluate_transaction

st.set_page_config(page_title="Fraud Risk Dashboard", layout="wide")

st.title("Bayesian Fraud Detection & Risk Scoring System")
st.write(
    "This dashboard estimates the **probability** that a transaction is "
    "fraudulent, then turns that probability into a business decision: "
    "**Allow, Manual Review, or Block.**"
)

# Load trained model + scaler
try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error(
        "No trained model found. Please run `python3 train_model.py` first, "
        "then restart this app."
    )
    st.stop()

df = load_data()
feature_cols = get_feature_columns(df)

st.sidebar.header("Enter Transaction Details")

# Let the user pick a sample row, type values one by one, or paste a full row at once
mode = st.sidebar.radio(
    "Choose input mode:",
    ["Enter manually", "Paste a full row"],
)

input_row = None  # will be filled by whichever mode runs below

if mode == "Enter manually":
    st.sidebar.write("Enter values for each feature:")
    values = {}
    for col in feature_cols:
        default_val = float(df[col].median())
        values[col] = st.sidebar.number_input(col, value=default_val)
    input_row = pd.DataFrame([values])[feature_cols]

else:  # "Paste a full row"
    st.sidebar.write(
        "Paste one row of comma-separated values, in this exact column order:"
    )
    st.sidebar.code(", ".join(feature_cols))

    example_row = df.iloc[0][feature_cols].tolist()
    example_text = ", ".join(str(v) for v in example_row)

    pasted_text = st.sidebar.text_area(
        "Paste values here (comma-separated)",
        placeholder=example_text,
        height=120,
    )

    if pasted_text.strip():
        try:
            # Split on commas (also works if the row was copied with spaces)
            raw_values = [v.strip() for v in pasted_text.strip().split(",")]
            numeric_values = [float(v) for v in raw_values]

            if len(numeric_values) != len(feature_cols):
                st.sidebar.error(
                    f"Expected {len(feature_cols)} values, got {len(numeric_values)}. "
                    "Please check your pasted row and try again."
                )
            else:
                input_row = pd.DataFrame([numeric_values], columns=feature_cols)
        except ValueError:
            st.sidebar.error(
                "Could not read the values. Make sure they are numbers separated by commas."
            )
    else:
        st.sidebar.info("Paste a row above to see the prediction. Showing the first dataset row for now.")
        input_row = df.iloc[[0]][feature_cols]

if input_row is None:
    st.stop()

amount = float(input_row["Amount"].values[0])

# Predict
input_scaled = scaler.transform(input_row)
fraud_probability = model.predict_proba(input_scaled)[0][1]

result = evaluate_transaction(fraud_probability, amount)

# ---- Display results ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud Probability", f"{result['fraud_probability']}%")
col2.metric("Risk Level", result["risk_level"])
col3.metric("Expected Loss", f"₹{result['expected_loss']:,.2f}")
col4.metric("Recommended Action", result["recommended_action"])

st.divider()

# ---- Probability gauge-style bar chart ----
st.subheader("Fraud Probability")
gauge_df = pd.DataFrame({
    "label": ["Fraud Probability"],
    "value": [result["fraud_probability"]],
})
fig = px.bar(
    gauge_df, x="value", y="label", orientation="h", range_x=[0, 100],
    text="value", color="value", color_continuous_scale=["green", "orange", "red"],
)
fig.update_layout(height=200, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---- Feature importance (only works for tree-based models) ----
st.subheader("What Influences This Model Most")
try:
    base_estimator = model.calibrated_classifiers_[0].estimator
    importances = base_estimator.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(10)

    fig2 = px.bar(importance_df, x="importance", y="feature", orientation="h")
    fig2.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)
except AttributeError:
    st.info("Feature importance is only available for tree-based models (Random Forest / XGBoost).")

st.divider()

# ---- Dataset overview ----
st.subheader("Dataset Overview")
fraud_counts = df["Class"].value_counts().rename({0: "Not Fraud", 1: "Fraud"})
fig3 = px.pie(values=fraud_counts.values, names=fraud_counts.index, title="Class Balance")
st.plotly_chart(fig3, use_container_width=True)
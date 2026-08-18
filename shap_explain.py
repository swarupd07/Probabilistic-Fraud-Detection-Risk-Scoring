# shap_explain.py

import numpy as np
import pandas as pd
import shap


def explain_transaction(model, background_data, input_scaled, feature_names):
    """
    Computes SHAP explanations for a single transaction.

    model:            fitted classifier with predict_proba
    background_data:  sample of scaled training rows used as reference
    input_scaled:     scaled feature row for the transaction (1 x n_features)
    feature_names:    list of column names
    """
    if hasattr(background_data, "values"):
        background_data = background_data.values
    if hasattr(input_scaled, "values"):
        input_scaled = input_scaled.values

    input_scaled = np.asarray(input_scaled)
    if input_scaled.ndim == 1:
        input_scaled = input_scaled.reshape(1, -1)

    max_evals = 2 * len(feature_names) + 1

    explainer = shap.Explainer(
        model.predict_proba, background_data, feature_names=feature_names
    )
    shap_result = explainer(input_scaled, max_evals=max_evals)
    return shap_result


def get_top_contributions(shap_result, feature_names, top_n=10):

    raw_values = shap_result.values

    # Handle various output tensor shapes from shap.Explainer
    if raw_values.ndim == 3:
        # Shape: (1, n_features, n_classes) -> take class 1 (Fraud)
        contributions = raw_values[0, :, 1] if raw_values.shape[2] > 1 else raw_values[0, :, 0]
    elif raw_values.ndim == 2:
        if raw_values.shape[0] == 1:
            contributions = raw_values[0]
        elif raw_values.shape[1] == 2:
            contributions = raw_values[:, 1]
        else:
            contributions = raw_values[0]
    elif raw_values.ndim == 1:
        contributions = raw_values
    else:
        contributions = np.ravel(raw_values)[:len(feature_names)]

    df = pd.DataFrame({
        "feature": list(feature_names),
        "contribution": contributions,
    })
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).head(top_n)
    df["direction"] = np.where(
        df["contribution"] >= 0, "Pushes toward Fraud", "Pushes toward Genuine"
    )

    return df.drop(columns="abs_contribution")

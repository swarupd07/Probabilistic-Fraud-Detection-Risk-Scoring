# app.py

# Streamlit dashboard for the Fraud Detection & Risk Scoring System.

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from utils import load_model, load_data, get_feature_columns
from decision_engine import (
    evaluate_transaction,
    get_thresholds,
    DEFAULT_BLOCK_THRESHOLD,
    REVIEW_RANGE,
)
from model_evaluation import get_precision_recall_curve, precision_at_k
from shap_explain import explain_transaction, get_top_contributions
from train_model import prepare_data

st.set_page_config(page_title="Fraud Risk Scoring Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Probabilistic Fraud Detection & Risk Scoring Engine")
st.write(
    "This dashboard predicts the **calibrated probability** that a transaction is fraudulent, "
    "and converts that probability into an automated risk decision: "
    "**Allow, Send for Manual Review, or Block.**"
)

# ---- Load trained model + scaler ----
try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ No trained model found. Please run `python3 train_model.py` first, "
        "then refresh this dashboard."
    )
    st.stop()

df = load_data()
feature_cols = get_feature_columns(df)


# ---------------------------------------------------------------------
# Reproduce the exact train/val/test split used during training
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading train/val/test splits...")
def get_split_data():
    (
        X_train_s,
        X_val_s,
        X_test_s,
        y_train,
        y_val,
        y_test,
        _scaler,
        X_train_raw,
        X_val_raw,
        X_test_raw,
    ) = prepare_data(df)
    return {
        "train": (X_train_raw, y_train),
        "val": (X_val_raw, y_val),
        "test": (X_test_raw, y_test),
    }


@st.cache_data(show_spinner="Computing performance metrics across splits...")
def compute_split_metrics(_model, _scaler, splits, block_thresh, review_thresh):
    rows = []
    split_probabilities = {}

    for split_name, (X_raw, y) in splits.items():
        X_scaled = _scaler.transform(X_raw)
        probs = _model.predict_proba(X_scaled)[:, 1]
        split_probabilities[split_name] = probs

        roc_auc = roc_auc_score(y, probs)
        pr_auc = average_precision_score(y, probs)

        preds_05 = (probs >= 0.5).astype(int)
        report_05 = classification_report(y, preds_05, output_dict=True, zero_division=0)

        preds_block = (probs >= block_thresh).astype(int)
        report_block = classification_report(y, preds_block, output_dict=True, zero_division=0)

        rows.append({
            "Split": split_name.capitalize(),
            "Total Rows": f"{len(y):,}",
            "Fraud Count": int(y.sum()),
            "Fraud Rate %": f"{100 * y.mean():.3f}%",
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Precision @0.5": round(report_05.get("1", {}).get("precision", 0.0), 3),
            "Recall @0.5": round(report_05.get("1", {}).get("recall", 0.0), 3),
            "F1 @0.5": round(report_05.get("1", {}).get("f1-score", 0.0), 3),
            f"Precision @Block({block_thresh:.2f})": round(report_block.get("1", {}).get("precision", 0.0), 3),
            f"Recall @Block({block_thresh:.2f})": round(report_block.get("1", {}).get("recall", 0.0), 3),
        })

    return pd.DataFrame(rows), split_probabilities


# ---------------------------------------------------------------------
# Sidebar: Threshold Controls & Transaction Input
# ---------------------------------------------------------------------
saved_block_thresh, saved_review_thresh = get_thresholds()

st.sidebar.header("⚙️ Decision Thresholds")
st.sidebar.caption("Optimal thresholds learned from validation cost minimization:")

block_threshold = st.sidebar.slider(
    "Block Threshold (High Risk)",
    min_value=0.05,
    max_value=0.99,
    value=float(saved_block_thresh),
    step=0.01,
    help="Transactions with fraud probability >= this cutoff will be immediately BLOCKED.",
)

review_threshold = st.sidebar.slider(
    "Review Threshold (Medium Risk)",
    min_value=0.01,
    max_value=float(block_threshold),
    value=float(min(saved_review_thresh, block_threshold)),
    step=0.01,
    help="Transactions between Review and Block cutoffs will be sent for MANUAL REVIEW.",
)

st.sidebar.markdown(
    f"""
    - **🔴 Block:** Prob ≥ `{block_threshold * 100:.1f}%`
    - **🟡 Review:** `{review_threshold * 100:.1f}%` ≤ Prob < `{block_threshold * 100:.1f}%`
    - **🟢 Allow:** Prob < `{review_threshold * 100:.1f}%`
    """
)

st.sidebar.divider()
st.sidebar.header("💳 Select or Enter Transaction")

input_mode = st.sidebar.radio(
    "Input Mode:",
    ["Dataset Presets", "Enter Manually", "Paste Row (CSV)"],
)

input_row = None

if input_mode == "Dataset Presets":
    preset_choice = st.sidebar.selectbox(
        "Choose an example transaction:",
        [
            "Sample Fraud Transaction #1",
            "Sample Fraud Transaction #2",
            "Sample Genuine Transaction #1",
            "Sample Genuine Transaction #2",
            "High Amount Genuine Transaction",
        ],
    )

    fraud_df = df[df["Class"] == 1]
    genuine_df = df[df["Class"] == 0]

    if preset_choice == "Sample Fraud Transaction #1":
        selected_idx = fraud_df.index[0]
    elif preset_choice == "Sample Fraud Transaction #2":
        selected_idx = fraud_df.index[1] if len(fraud_df) > 1 else fraud_df.index[0]
    elif preset_choice == "Sample Genuine Transaction #1":
        selected_idx = genuine_df.index[0]
    elif preset_choice == "Sample Genuine Transaction #2":
        selected_idx = genuine_df.index[1] if len(genuine_df) > 1 else genuine_df.index[0]
    else:
        selected_idx = genuine_df.sort_values("Amount", ascending=False).index[0]

    input_row = df.loc[[selected_idx], feature_cols]
    actual_label = "Fraud" if df.loc[selected_idx, "Class"] == 1 else "Genuine"
    st.sidebar.info(f"Loaded Row #{selected_idx} | Ground Truth: **{actual_label}** | Amount: ₹{input_row['Amount'].values[0]:,.2f}")

elif input_mode == "Enter Manually":
    st.sidebar.write("Enter feature values:")
    values = {}
    with st.sidebar.expander("Key Features (Time & Amount)", expanded=True):
        values["Time"] = st.number_input("Time (seconds)", value=float(df["Time"].median()))
        values["Amount"] = st.number_input("Amount (₹)", value=float(df["Amount"].median()), min_value=0.0)

    with st.sidebar.expander("PCA Features (V1 - V28)", expanded=False):
        for col in [c for c in feature_cols if c not in ["Time", "Amount"]]:
            values[col] = st.number_input(col, value=float(df[col].median()), format="%.4f")

    input_row = pd.DataFrame([values])[feature_cols]

else:  # "Paste Row (CSV)"
    st.sidebar.write("Paste 30 comma-separated numeric values:")
    st.sidebar.code(", ".join(feature_cols))

    example_row = df.iloc[0][feature_cols].tolist()
    example_text = ", ".join(f"{v:.4f}" for v in example_row)

    pasted_text = st.sidebar.text_area(
        "Paste values here",
        placeholder=example_text,
        height=100,
    )

    if pasted_text.strip():
        try:
            raw_values = [v.strip() for v in pasted_text.strip().split(",")]
            numeric_values = [float(v) for v in raw_values]

            if len(numeric_values) != len(feature_cols):
                st.sidebar.error(
                    f"Expected {len(feature_cols)} values, but got {len(numeric_values)}."
                )
            else:
                input_row = pd.DataFrame([numeric_values], columns=feature_cols)
        except ValueError:
            st.sidebar.error("Could not parse numbers. Ensure values are comma-separated floats.")
    else:
        input_row = df.iloc[[0]][feature_cols]

if input_row is None:
    st.stop()


# ---------------------------------------------------------------------
# Score Transaction
# ---------------------------------------------------------------------
amount = float(input_row["Amount"].values[0])
input_scaled = scaler.transform(input_row)
fraud_probability = float(model.predict_proba(input_scaled)[0, 1])

result = evaluate_transaction(
    fraud_probability,
    amount,
    block_threshold=block_threshold,
    review_threshold=review_threshold,
)


# ---------------------------------------------------------------------
# Section 1: Transaction Risk Assessment Card
# ---------------------------------------------------------------------
st.header("🎯 Real-Time Transaction Assessment")

col1, col2, col3, col4 = st.columns(4)

action_colors = {
    "Allow": "🟢",
    "Send for Manual Review": "🟡",
    "Block Transaction": "🔴",
}
action_icon = action_colors.get(result["recommended_action"], "⚪")

col1.metric("Predicted Fraud Risk", f"{result['fraud_probability']:.2f}%")
col2.metric("Risk Tier", result["risk_level"])
col3.metric("Expected Loss", f"₹{result['expected_loss']:,.2f}")
col4.metric("Recommended Action", f"{action_icon} {result['recommended_action']}")

# Decision Gauge / Risk Bar
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=result["fraud_probability"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Fraud Probability Score & Decision Bands (%)", "font": {"size": 16}},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1f77b4", "thickness": 0.3},
            "steps": [
                {"range": [0, review_threshold * 100], "color": "#d4edda"},      # Green (Allow)
                {"range": [review_threshold * 100, block_threshold * 100], "color": "#fff3cd"},  # Yellow (Review)
                {"range": [block_threshold * 100, 100], "color": "#f8d7da"},    # Red (Block)
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": block_threshold * 100,
            },
        },
    )
)
fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------
# Section 2: Model Interpretability (Global & Local SHAP)
# ---------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Global Model Feature Importance")
    st.caption("Which features contribute most to the model's global fraud risk calculation.")

    # Extract base estimator safely (whether calibrated or direct)
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        calib = model.calibrated_classifiers_[0]
        base_estimator = getattr(calib, "estimator", getattr(calib, "base_estimator", None))
    else:
        base_estimator = model

    if base_estimator is not None and hasattr(base_estimator, "feature_importances_"):
        # Tree-based model (Random Forest / XGBoost)
        importances = base_estimator.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(10)

        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 10 Feature Importances ({type(base_estimator).__name__})",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    elif base_estimator is not None and hasattr(base_estimator, "coef_"):
        # Linear model (Logistic Regression)
        coefs = base_estimator.coef_[0]
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": coefs,
            "Abs_Weight": np.abs(coefs),
            "Impact": np.where(coefs >= 0, "Increases Fraud Risk", "Decreases Fraud Risk"),
        }).sort_values("Abs_Weight", ascending=False).head(10)

        fig_imp = px.bar(
            imp_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title=f"Top 10 Standardized Coefficients ({type(base_estimator).__name__})",
            color="Impact",
            color_discrete_map={
                "Increases Fraud Risk": "#e74c3c",
                "Decreases Fraud Risk": "#2ecc71",
            },
        )
        fig_imp.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not directly exposed by current estimator.")

with col_right:
    st.subheader("🔍 Local Transaction Explanation (SHAP)")
    st.caption("Feature-by-feature explanation for THIS specific transaction.")

    try:
        bg_sample = df[feature_cols].sample(n=min(50, len(df)), random_state=42)
        bg_scaled = scaler.transform(bg_sample)

        with st.spinner("Calculating SHAP attribution..."):
            shap_res = explain_transaction(model, bg_scaled, input_scaled, feature_cols)
            top_shap = get_top_contributions(shap_res, feature_cols, top_n=10)

        fig_shap = px.bar(
            top_shap,
            x="contribution",
            y="feature",
            orientation="h",
            title="Top 10 SHAP Feature Contributions",
            color="direction",
            color_discrete_map={
                "Pushes toward Fraud": "#e74c3c",
                "Pushes toward Genuine": "#2ecc71",
            },
        )
        fig_shap.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute SHAP explanation: {e}")

st.divider()


# ---------------------------------------------------------------------
# Section 3: Performance Across Train / Validation / Test
# ---------------------------------------------------------------------
st.header("📈 Model Evaluation & Stability Across Data Splits")
st.caption(
    "**Train:** Learned representations. **Validation:** Threshold optimization and calibration tuning. "
    "**Test:** Completely untouched holdout set providing unbiased generalization metrics."
)

splits = get_split_data()
metrics_df, split_probs = compute_split_metrics(
    model, scaler, splits, block_threshold, review_threshold
)

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# PR Curve & Precision@K for Test split
with st.expander("📊 View Precision-Recall Curve & Precision@K (Test Holdout Split)", expanded=True):
    col_pr, col_k = st.columns([3, 2])

    test_probs = split_probs["test"]
    test_labels = splits["test"][1].values

    precision, recall, _ = get_precision_recall_curve(test_labels, test_probs)
    pr_plot_df = pd.DataFrame({"Recall": recall, "Precision": precision})

    with col_pr:
        fig_pr = px.line(
            pr_plot_df,
            x="Recall",
            y="Precision",
            title=f"Precision-Recall Curve (Holdout Test Set | PR-AUC: {average_precision_score(test_labels, test_probs):.4f})",
        )
        fig_pr.update_layout(height=340)
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_k:
        k_percentages = [0.5, 1.0, 2.0, 5.0, 10.0]
        k_data = []
        for k in k_percentages:
            p_val = precision_at_k(test_labels, test_probs, k_percent=k)
            n_reviewed = int(np.ceil(len(test_labels) * (k / 100)))
            k_data.append({
                "Top Flagged %": f"Top {k}%",
                "Transactions Reviewed": f"{n_reviewed:,}",
                "Precision": f"{p_val:.3f}",
            })
        st.write("**Precision@K (Investigative Capacity)**")
        st.caption("If operations team reviews top K% riskiest transactions:")
        st.dataframe(pd.DataFrame(k_data), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------
# Section 4: Dataset Summary
# ---------------------------------------------------------------------
with st.expander("📁 Dataset Summary & Imbalance Overview"):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        class_counts = df["Class"].value_counts().rename({0: "Genuine (0)", 1: "Fraud (1)"})
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Class Balance",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_d2:
        st.write("**Summary Statistics:**")
        st.write(f"- Total Transactions: **{len(df):,}**")
        st.write(f"- Genuine Transactions: **{(df['Class'] == 0).sum():,}** ({(df['Class'] == 0).mean() * 100:.2f}%)")
        st.write(f"- Fraud Transactions: **{(df['Class'] == 1).sum():,}** ({(df['Class'] == 1).mean() * 100:.4f}%)")
        st.write(f"- Mean Transaction Amount: **₹{df['Amount'].mean():,.2f}**")
        st.write(f"- Max Transaction Amount: **₹{df['Amount'].max():,.2f}**")

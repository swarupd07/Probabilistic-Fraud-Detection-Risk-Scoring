# train_model.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

from utils import (
    load_data,
    get_feature_columns,
    save_model,
    save_threshold,
)

from model_evaluation import (
    compare_calibration_methods,
    print_precision_at_k_table,
    find_cost_optimal_threshold,
)


def prepare_data(df, temporal_split=True):
    feature_cols = get_feature_columns(df)

    if temporal_split:
        df_sorted = df.sort_values("Time").reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.85)
        val_split_idx = int(len(df_sorted) * 0.70)
        train_df = df_sorted.iloc[:val_split_idx]
        val_df = df_sorted.iloc[val_split_idx:split_idx]
        test_df = df_sorted.iloc[split_idx:]
    else:
        # Stratified 70% train, 15% validation, 15% test
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df["Class"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42, stratify=temp_df["Class"]
        )

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["Class"]
    y_val = val_df["Class"]
    y_test = test_df["Class"]

    # Scaling helps Logistic Regression converge faster and fairly.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        X_train,
        X_val,
        X_test,
    )


def evaluate_model(name, model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print(f"\n--- {name} ---")
    print(classification_report(y_val, preds, digits=3, zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(y_val, probs), 4))
    print("Average Precision (PR-AUC):", round(average_precision_score(y_val, probs), 4))

    return average_precision_score(y_val, probs)  # Using PR-AUC for imbalanced data


def train_all_models():
    df = load_data()
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        X_train_raw,
        X_val_raw,
        X_test_raw,
    ) = prepare_data(df)

    # print_psi_report(X_train_raw, X_test_raw, columns=["Amount"])

    # Over-sample the minority class in the training split only
    '''smote = SMOTE(sampling_strategy=0.30, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(
        f"\nAfter SMOTE: {len(y_train_resampled)} training rows "
        f"({(y_train_resampled == 1).sum()} fraud, {(y_train_resampled == 0).sum()} genuine)"
    )'''

    os.makedirs("models", exist_ok=True)

    # 1. Logistic Regression baseline
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    score_lr = evaluate_model("Logistic Regression", log_reg, X_val, y_val)

    # 2. Random Forest baseline
    rf = RandomForestClassifier(
        n_estimators=100,  random_state=42, n_jobs=-1, max_depth=7, class_weight="balanced")
    rf.fit(X_train, y_train)
    score_rf = evaluate_model("Random Forest", rf, X_val, y_val)

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    # 3. XGBoost baseline
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=ratio
    )
    xgb.fit(X_train, y_train)
    score_xgb = evaluate_model("XGBoost", xgb, X_val, y_val)

    # Best model by PR-AUC on the validation set
    scores = {
        "Logistic Regression": (score_lr, log_reg),
        "Random Forest": (score_rf, rf),
        "XGBoost": (score_xgb, xgb),
    }
    best_name = max(scores, key=lambda k: scores[k][0])
    best_model = scores[best_name][1]
    print(f"\nBest model based on PR-AUC: {best_name}")

    # Calibration comparison (Sigmoid vs Isotonic)
    calibrated_model, chosen_method, calibration_results = compare_calibration_methods(
        best_model, X_train, y_train, X_val, y_val
    )
    evaluate_model(
        f"{best_name} + Calibration ({chosen_method})", calibrated_model, X_val, y_val
    )

    final_probs = calibrated_model.predict_proba(X_val)[:, 1]

    # Precision@K: business-relevant operational metric
    print_precision_at_k_table(y_val.values, final_probs)

    # Cost-based threshold optimization
    best_threshold, best_cost, cost_curve = find_cost_optimal_threshold(
        y_val.values, final_probs, X_val_raw["Amount"].values
    )

    # Derive review threshold (5% below block threshold, bounded at 0.0)
    review_threshold = round(max(0.0, best_threshold - 0.05), 2)

    save_threshold(best_threshold, review_threshold)
    save_model(calibrated_model, scaler)

    return calibrated_model, scaler


if __name__ == "__main__":
    train_all_models()

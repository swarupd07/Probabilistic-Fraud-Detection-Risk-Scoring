# train_model.py


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

from utils import load_data, get_feature_columns, save_model

def prepare_data(df):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Class"]

    # Scaling helps Logistic Regression converge faster and fairly.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def evaluate_model(name, model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, digits=3, zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print("Average Precision (PR-AUC):", round(average_precision_score(y_test, probs), 4))

    return roc_auc_score(y_test, probs)


def train_all_models():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # 1. Logistic Regression baseline
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    score_lr = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

    # 2. Random Forest baseline
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    score_rf = evaluate_model("Random Forest", rf, X_test, y_test)

    # 3. XGBoost baseline
    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=fraud_ratio,  # handles class imbalance
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    score_xgb = evaluate_model("XGBoost", xgb, X_test, y_test)

    # best model by ROC-AUC
    scores = {"Logistic Regression": (score_lr, log_reg),
              "Random Forest": (score_rf, rf),
              "XGBoost": (score_xgb, xgb)}
    best_name = max(scores, key=lambda k: scores[k][0])
    best_model = scores[best_name][1]
    print(f"\nBest model based on ROC-AUC: {best_name}")

    # ---- Bayesian-style probability calibration ----
    ''' This wraps the best model and corrects its output probabilities
     so that "0.8 probability" really does mean "fraud about 80% of
     the time" on unseen data. This is what makes the probabilities
     trustworthy enough to use for expected-loss calculations later.'''
    
    print("\nCalibrating probabilities (isotonic method)...")
    calibrated_model = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated_model.fit(X_train, y_train)
    evaluate_model(f"{best_name} + Calibration", calibrated_model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    save_model(calibrated_model, scaler)

    return calibrated_model, scaler


"""if __name__ == "__main__":
    train_all_models()"""

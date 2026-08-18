# utils.py

import os
import json
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "optimal_threshold.json")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please place creditcard.csv in data/ directory.")
    df = pd.read_csv(DATA_PATH)
    return df


def get_feature_columns(df):
    return [col for col in df.columns if col != "Class"]


def save_model(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Model or Scaler not found in {MODEL_DIR}. Please run train_model.py first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def save_threshold(block_threshold, review_threshold=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {"block_threshold": float(block_threshold)}
    if review_threshold is not None:
        payload["review_threshold"] = float(review_threshold)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Optimal threshold saved to {THRESHOLD_PATH}")


def load_threshold(default=0.65):
    # Returns the saved cost-optimal block threshold, or `default` if none exists yet
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH) as f:
                data = json.load(f)
            return float(data.get("block_threshold", default))
        except (json.JSONDecodeError, ValueError, TypeError):
            return default
    return default


def load_thresholds(default_block=0.65, default_review=0.60):
    # Returns a tuple of (block_threshold, review_threshold)
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH) as f:
                data = json.load(f)
            block = float(data.get("block_threshold", default_block))
            review = float(data.get("review_threshold", max(0.0, block - 0.05)))
            return block, review
        except (json.JSONDecodeError, ValueError, TypeError):
            return default_block, default_review
    return default_block, default_review

# utils.py

import pandas as pd
import joblib

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def get_feature_columns(df):
    #All columns except the label 'Class' are features.
    return [col for col in df.columns if col != "Class"]


def save_model(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

import pandas as pd
import numpy as np
import joblib

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features


# -----------------------------
# Load Saved Artifacts
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
selected_features = joblib.load("models/features.pkl")


# -----------------------------
# Predict Function
# -----------------------------
def predict_gdp(input_df: pd.DataFrame):

    # Apply feature engineering
    X, _, _, _ = prepare_features(input_df)

    # Select same features
    X = X[selected_features]

    # Predict log GDP
    y_pred_log = model.predict(X)

    # Convert back to original GDP scale
    y_pred = np.expm1(y_pred_log)

    return y_pred

def simulate_scenario(df: pd.DataFrame, changes: dict, selected_features: list) -> pd.DataFrame:
    df_copy = df.copy()

    for feature, change in changes.items():
        # Apply only if feature exists in original data
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature] * (1 + change)

    return df_copy

def predict_scenario(df: pd.DataFrame, changes: dict):
    # baseline
    base_pred = predict_gdp(df)

    # modified data
    df_new = simulate_scenario(df, changes)
    new_pred = predict_gdp(df_new)

    return base_pred, new_pred
# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    df = preprocess_pipeline("data/raw/World_data_GDP.csv")
    sample = df.sample(3, random_state=42)

    scenario = {
        "Exports of goods and services (current US$)": 0.05,
        "Imports of goods and services (current US$)": -0.02
    }

    base, new = predict_scenario(sample, scenario)

    print("\nBaseline GDP:", base)
    print("Scenario GDP:", new)
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
    print("\n📥 Input to predict_gdp():")
    print(input_df.head())
    # Apply feature engineering
    X, _, _, _ = prepare_features(input_df)
    print("\n🧠 Features after engineering:")
    print(X.head())
    # Select same features
    X = X[selected_features]
    print("\n🎯 Features used for prediction:")
    print(X.head())
    # Predict log GDP
    y_pred_log = model.predict(X)

    # Convert back to original GDP scale
    y_pred = np.expm1(y_pred_log)

    return y_pred

def simulate_scenario(df: pd.DataFrame, changes: dict):
    df_copy = df.copy()
    print("\n⚙️ Applying Scenario Changes:")
    print("Changes:", changes)
    for feature, change in changes.items():
        # Apply only if feature exists in original data
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature] * (1 + change)
    print("\n📊 Modified Data (after scenario):")
    print(df_copy.head())
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
    print("\n🔍 Difference:")
    print(new - base)
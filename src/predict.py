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


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    # Example: use last few rows
    df = preprocess_pipeline("data/raw/World_data_GDP.csv")

    sample = df.sample(5, random_state=42)

    predictions = predict_gdp(sample)

    print("Predicted GDP values:")
    print(predictions)
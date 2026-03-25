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

def predict_gdp_from_features(X):
    y_pred_log = model.predict(X)
    return np.expm1(y_pred_log)

def explain_scenario(X, X_new):
    diff = X_new - X

    print("\n📊 Feature Impact (Change):")
    print(diff)

    print("\n📈 Absolute Impact Ranking:")
    impact = diff.abs().mean().sort_values(ascending=False)
    print(impact)
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

    # Step 1: Convert raw data → features
    X, _, _, _ = prepare_features(df)

    print("\n🧠 Original Features:")
    print(X.head())

    # Step 2: Select trained features
    X = X[selected_features]

    # Step 3: Baseline prediction
    base_pred = predict_gdp_from_features(X)

    # Step 4: Apply scenario DIRECTLY on features
    X_new = X.copy()

    print("\n⚙️ Applying Scenario on Features:")
    print("Changes:", changes)

    # Step 4: Apply scenario on RAW data
    df_new = df.copy()

    print("\n⚙️ Applying Scenario on RAW data:")

    for raw_feature, change in {
        "Exports of goods and services (current US$)": changes.get(
            "economic__Exports_of_goods_and_services_current_USusd", 0
        ),
        "Imports of goods and services (current US$)": changes.get(
            "economic__Imports_of_goods_and_services_current_USusd", 0
        )
    }.items():
        
        if raw_feature in df_new.columns:
            df_new[raw_feature] = df_new[raw_feature] * (1 + change)

    # Recompute features AFTER change
    X_new, _, _, _ = prepare_features(df_new)
    X_new = X_new[selected_features] * (1 + change * 5)
    
    # 🔍 Explain impact
    explain_scenario(X, X_new)

    # =============================
    # 🔍 ECONOMIC CONSISTENCY CHECK
    # =============================
    print("\n🧠 Economic Consistency Check:")

    for col in ["trade_contribution", "economic_activity", "gdp_proxy_signal"]:
        if col in X_new.columns:
            print(f"{col}:")
            print(X_new[col].values)

    print("\n📊 Modified Features:")
    print(X_new.head())

    # Step 5: New prediction
    new_pred = predict_gdp_from_features(X_new)

    return base_pred, new_pred

def run_sensitivity(df, feature):

    values = [-0.2, -0.1, 0, 0.1, 0.2]

    print(f"\n📊 Sensitivity Analysis for {feature}")

    for v in values:
        scenario = {feature: v}
        base, new = predict_scenario(df, scenario)

        print(f"Change {v*100:+.0f}% → ΔGDP = {(new - base).mean():.2e}")
# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    df = preprocess_pipeline("data/raw/World_data_GDP.csv")
    sample = df.sample(3, random_state=42)

    scenario = {
    "economic__Exports_of_goods_and_services_current_USusd": 0.05,
    "economic__Imports_of_goods_and_services_current_USusd": -0.02
}
    
    base, new = predict_scenario(sample, scenario)

    print("\nBaseline GDP:", base)
    print("Scenario GDP:", new)
    print("\n🔍 Difference:")
    print(new - base)
    run_sensitivity(
    sample,
    "economic__Exports_of_goods_and_services_current_USusd"
)
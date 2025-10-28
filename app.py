import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# 📦 Load saved components
# ----------------------------
models_dir = "models"

encoder = joblib.load(f"{models_dir}/encoder.pkl")
columns = joblib.load(f"{models_dir}/X_columns.pkl")
scaler = joblib.load(f"{models_dir}/scaling_pipeline.pkl")
df_imputed = joblib.load(f"{models_dir}/df_imputed.pkl")
metrics_df = pd.read_csv(f"{models_dir}/model_metrics.csv")

# Load all trained models dynamically
models = {}
for file in os.listdir(models_dir):
    if (
        file.endswith(".pkl")
        and not file.startswith(("encoder", "X_columns", "df_imputed", "scaling_pipeline"))
    ):
        name = file.replace(".pkl", "").replace("_", " ")
        models[name] = joblib.load(os.path.join(models_dir, file))

st.set_page_config(page_title="Global GDP Predictor", page_icon="🌍", layout="wide")
st.title("🌍 Global GDP Predictor Dashboard")
st.caption("Compare GDP Predictions Across Multiple Models")

# ----------------------------
# 🌐 Select Country & Modify Indicators
# ----------------------------
country = st.selectbox("Select Country:", sorted(df_imputed["Country Name"].unique()))
country_data = df_imputed[df_imputed["Country Name"] == country].iloc[-1].drop(["GDP (current US$)"], errors="ignore")

st.subheader("🔧 Modify Key Economic Indicators")
cols = st.columns(2)

# Editable inputs for first few indicators (to keep UI compact)
for i, col_name in enumerate(country_data.index[:10]):  # show first 10 numeric indicators
    with cols[i % 2]:
        if np.issubdtype(type(country_data[col_name]), np.number):
            country_data[col_name] = st.number_input(
                f"{col_name}", value=float(country_data[col_name])
            )

# ----------------------------
# 🧮 Preprocessing before Prediction
# ----------------------------
# Encode the country name
encoded_country = pd.DataFrame(
    encoder.transform([[country]]),
    columns=encoder.get_feature_names_out(["Country Name"]),
)

# Apply same scaling as training
numeric_cols = [
    c for c in df_imputed.columns if c not in ["Country Name", "GDP (current US$)"]
]

to_scale = pd.DataFrame([country_data[numeric_cols]])
scaled_array = scaler.transform(to_scale)
scaled_df = pd.DataFrame(
    scaled_array, columns=scaler.get_feature_names_out(), index=[0]
)

# Combine scaled indicators + encoded country dummies
country_input = pd.concat([scaled_df, encoded_country], axis=1)
country_input.columns = country_input.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True)
country_input = country_input.reindex(columns=columns, fill_value=0)

# ----------------------------
# 📈 Predict GDP with each model
# ----------------------------
st.subheader("📊 Model Predictions")
results = []

for name, model in models.items():
    try:
        pred = model.predict(country_input)[0]
        metrics_row = metrics_df.loc[metrics_df["Model"] == name.replace("_", " ")]
        metrics = metrics_row.to_dict("records")[0] if not metrics_row.empty else {}
        results.append({
            "Model": name,
            "Predicted GDP (US$)": f"${pred:,.2f}",
            **metrics
        })
    except Exception as e:
        results.append({"Model": name, "Predicted GDP (US$)": "⚠️ Error", "Error": str(e)})

pred_df = pd.DataFrame(results)
st.dataframe(pred_df.style.format(precision=2))

# ----------------------------
# 🏆 Highlight Best Model
# ----------------------------
if not pred_df.empty and "R2" in pred_df.columns:
    best = pred_df.sort_values("R2", ascending=False).iloc[0]
    st.success(
        f"🏆 **Best Model:** {best['Model']} | R² = {best['R2']} | Predicted GDP = {best['Predicted GDP (US$)']}"
    )

# ----------------------------
# 📉 Baseline Comparison
# ----------------------------
st.subheader("📉 Compare With Baseline (Unmodified Indicators)")

# Baseline input (without user edits)
baseline_row = df_imputed[df_imputed["Country Name"] == country].iloc[-1].drop(["GDP (current US$)"], errors="ignore")
to_scale_base = pd.DataFrame([baseline_row[numeric_cols]])
scaled_base = scaler.transform(to_scale_base)
scaled_base_df = pd.DataFrame(scaled_base, columns=scaler.get_feature_names_out())
baseline_input = pd.concat([scaled_base_df, encoded_country], axis=1)
baseline_input.columns = baseline_input.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True)
baseline_input = baseline_input.reindex(columns=columns, fill_value=0)

# Predict baseline GDP with best model
best_model_name = best["Model"]
best_model = models[best_model_name]
base_pred = best_model.predict(baseline_input)[0]
new_pred = float(pred_df.loc[pred_df["Model"] == best_model_name, "Predicted GDP (US$)"].iloc[0].replace("$", "").replace(",", ""))

change = ((new_pred - base_pred) / base_pred) * 100
st.write(f"**Predicted GDP Change:** {change:+.2f}%")

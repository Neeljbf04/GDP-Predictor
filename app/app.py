import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from src.predict import predict_scenario

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="GDP Predictor", layout="wide")

st.title("🌍 GDP Prediction Dashboard")
st.markdown("Upload a dataset to predict GDP values using a trained ML model.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
st.subheader("⚙️ Scenario Controls")

# -----------------------------
# Available Features for Scenario
# -----------------------------
all_features = [
    "economic__Exports_of_goods_and_services_current_USusd",
    "economic__Imports_of_goods_and_services_current_USusd",
    "economic__Gross_capital_formation_current_USusd",
    "social__Individuals_using_the_Internet_percent_of_population",
    "social__Urban_population",
    "economic__Agriculture_forestry_and_fishing_value_added_current_USusd",
    "economic__Total_reserves_includes_gold_current_USusd"
]
selected_features = st.multiselect(
    "🎯 Select Features to Modify",
    all_features,
    default=[
        "economic__Exports_of_goods_and_services_current_USusd",
        "economic__Imports_of_goods_and_services_current_USusd"
    ]
)
scenario = {}

st.markdown("### 🎛️ Adjust Feature Changes (%)")

for feature in selected_features:
    change = st.slider(f"{feature}", -100, 100, 0)
    scenario[feature] = change / 100

if uploaded_file:
    df = load_data(uploaded_file)
    # 🔥 Basic cleaning for user input
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    try:
        with st.spinner("Running prediction..."):
            base, new = predict_scenario(df, scenario)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Predictions
    # -----------------------------
    base, new = predict_scenario(df, scenario)

    result_df = df.copy()
    result_df["Baseline GDP"] = base
    result_df["Scenario GDP"] = new
    result_df["GDP Change"] = new - base

    # -----------------------------
    # Format GDP (Readable)
    # -----------------------------
    def format_gdp(x):
        if x >= 1e12:
            return f"${x/1e12:.2f} Trillion"
        elif x >= 1e9:
            return f"${x/1e9:.2f} Billion"
        else:
            return f"${x:,.0f}"

    result_df["Formatted GDP"] = result_df["Scenario GDP"].apply(format_gdp)

    # -----------------------------
    # KPI Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("🌍 Avg Baseline GDP", format_gdp(np.mean(base)))
    col2.metric("🚀 Avg Scenario GDP", format_gdp(np.mean(new)))
    col3.metric("📊 Avg Change", format_gdp(np.mean(new - base)))

    # -----------------------------
    # Results Table
    # -----------------------------
    st.subheader("📊 Scenario Results")

    display_cols = ["Formatted GDP"]

    if "Country Name" in result_df.columns:
        display_cols = ["Country Name"]

    display_df = result_df.copy()

    display_df["Baseline GDP"] = display_df["Baseline GDP"].apply(format_gdp)
    display_df["Scenario GDP"] = display_df["Scenario GDP"].apply(format_gdp)
    display_df["Change"] = display_df["GDP Change"].apply(format_gdp)

    if "Country Name" in result_df.columns:
        st.dataframe(display_df[["Country Name", "Baseline GDP", "Scenario GDP", "Change"]])
    else:
        st.dataframe(display_df[["Baseline GDP", "Scenario GDP", "Change"]])

    # -----------------------------
    # Chart
    # -----------------------------
    st.subheader("🌍 Country-wise GDP Comparison")

    display_df = result_df.copy()

    display_df["Baseline GDP"] = display_df["Baseline GDP"].apply(format_gdp)
    display_df["Scenario GDP"] = display_df["Scenario GDP"].apply(format_gdp)
    display_df["Change"] = display_df["GDP Change"].apply(format_gdp)

    st.dataframe(display_df[[
        "Country Name",
        "Baseline GDP",
        "Scenario GDP",
        "Change"
    ]])

    st.subheader("📊 Country-wise GDP Change")

    chart_df = pd.DataFrame({
        "Country": result_df["Country Name"],
        "Change": result_df["GDP Change"]
    }).set_index("Country")

    st.bar_chart(chart_df)

    top_gainers = result_df.sort_values("GDP Change", ascending=False).head(5)
    top_losers = result_df.sort_values("GDP Change").head(5)

    st.subheader("🚀 Top Gainers")
    st.dataframe(top_gainers[["Country Name", "GDP Change"]])

    st.subheader("📉 Top Losers")
    st.dataframe(top_losers[["Country Name", "GDP Change"]])
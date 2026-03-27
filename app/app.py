import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from src.predict import predict_scenario

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

exports_change = st.slider("Exports Change (%)", -20, 20, 0)
imports_change = st.slider("Imports Change (%)", -20, 20, 0)
investment_change = st.slider("Investment Change (%)", -20, 20, 0)
internet_change = st.slider("Internet Usage Change (%)", -20, 20, 0)

scenario = {
    "economic__Exports_of_goods_and_services_current_USusd": exports_change / 100,
    "economic__Imports_of_goods_and_services_current_USusd": imports_change / 100,
    "economic__Gross_capital_formation_current_USusd": investment_change / 100,
    "social__Individuals_using_the_Internet_percent_of_population": internet_change / 100
}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # 🔥 Basic cleaning for user input
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))

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
    st.subheader("📈 GDP Comparison")

    chart_df = pd.DataFrame({
        "Baseline": base,
        "Scenario": new
    })

    if "Country Name" in result_df.columns:
        chart_df.index = result_df["Country Name"]

    st.bar_chart(chart_df)
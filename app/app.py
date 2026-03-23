import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

from src.predict import predict_gdp

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

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Predictions
    # -----------------------------
    predictions = predict_gdp(df)

    result_df = df.copy()
    result_df["Predicted GDP"] = predictions

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

    result_df["Formatted GDP"] = result_df["Predicted GDP"].apply(format_gdp)

    # -----------------------------
    # KPI Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("🌍 Avg GDP", format_gdp(np.mean(predictions)))
    col2.metric("📈 Max GDP", format_gdp(np.max(predictions)))
    col3.metric("📉 Min GDP", format_gdp(np.min(predictions)))

    # -----------------------------
    # Results Table
    # -----------------------------
    st.subheader("📊 Prediction Results")

    if "Country Name" in result_df.columns:
        st.dataframe(result_df[["Country Name", "Formatted GDP"]])
    else:
        st.dataframe(result_df[["Formatted GDP"]])

    # -----------------------------
    # Chart
    # -----------------------------
    st.subheader("📈 GDP Distribution")

    chart_df = result_df.copy()

    if "Country Name" in chart_df.columns:
        chart_df = chart_df.set_index("Country Name")

    st.bar_chart(chart_df["Predicted GDP"])
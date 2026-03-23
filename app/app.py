import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from src.predict import predict_gdp
from src.data_preprocessing import preprocess_pipeline

st.title("🌍 GDP Predictor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    predictions = predict_gdp(df)

    st.write("### Predictions")
    st.write(predictions)
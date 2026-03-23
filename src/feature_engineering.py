import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple,cast


# -----------------------------
# 1. Target Transformation
# -----------------------------
def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log transform GDP
    df["GDP_log"] = np.log1p(df["GDP (current US$)"])

    return df


# -----------------------------
# 2. Encode Country
# -----------------------------

def encode_country(df: pd.DataFrame) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    df = df.copy()
    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(df[["Country Name"]])
    encoded = encoded.astype(float)  # ensure numeric

    df["Country_Code"] = pd.Series(encoded[:, 0], index=df.index)
    return df, encoder


# -----------------------------
# 3. Feature Groups
# -----------------------------
def get_feature_groups():
    group_economic = [
        'Exports of goods and services (current US$)',
        'Imports of goods and services (current US$)',
        'External debt stocks, total (DOD, current US$)',
        'Gross capital formation (current US$)',
        'General government final consumption expenditure (current US$)',
        'Services, value added (current US$)',
        'Industry (including construction), value added (current US$)',
        'Agriculture, forestry, and fishing, value added (current US$)',
        'Total reserves (includes gold, current US$)'
    ]

    group_percentage = [
        'GDP growth (annual %)',
        'Inflation, consumer prices (annual %)',
        'Real interest rate (%)',
        'Unemployment, total (% of total labor force)',
        'Gross savings (% of GDP)',
        'Foreign direct investment, net inflows (% of GDP)'
    ]

    group_social = [
        'Population, total',
        'Urban population',
        'Life expectancy at birth, total (years)',
        'Individuals using the Internet (% of population)'
    ]

    group_financial = [
        'Foreign direct investment, net inflows (BoP, current US$)',
        'Current account balance (BoP, current US$)',
        'Official exchange rate (LCU per US$, period average)'
    ]

    return group_economic, group_percentage, group_social, group_financial


# -----------------------------
# 4. Scaling Pipeline
# -----------------------------
def build_scaling_pipeline(df: pd.DataFrame):
    group_economic, group_percentage, group_social, group_financial = get_feature_groups()

    def filter_existing(columns, df):
        return [col for col in columns if col in df.columns]


    group_economic = filter_existing(group_economic, df)
    group_percentage = filter_existing(group_percentage, df)
    group_social = filter_existing(group_social, df)
    group_financial = filter_existing(group_financial, df)


    column_transformer = ColumnTransformer(transformers=[
        ('economic', PowerTransformer(), group_economic),
        ('percentage', StandardScaler(), group_percentage),
        ('social', MinMaxScaler(), group_social),
        ('financial', RobustScaler(), group_financial)
    ], remainder='drop')

    pipeline = Pipeline([
        ('scaler', column_transformer)
    ])

    return pipeline


# -----------------------------
# 5. Prepare Final Features
# -----------------------------
def prepare_features(df: pd.DataFrame):
    df = transform_target(df)

    df, encoder = encode_country(df)
    df = cast(pd.DataFrame, df)

    # Separate target
    y = df["GDP_log"]

    # Drop unused columns
    X = df.drop(columns=[
        "GDP (current US$)",
        "GDP_log",
        "Country Name"
    ]).copy()
    X=cast(pd.DataFrame, X)

    # Build scaling pipeline
    scaler = build_scaling_pipeline(df)

    X_scaled = scaler.fit_transform(X)

    feature_names = scaler.get_feature_names_out()

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Add country code back
    X_scaled_df["Country_Code"] = df["Country_Code"].values

    return X_scaled_df, y, scaler, encoder
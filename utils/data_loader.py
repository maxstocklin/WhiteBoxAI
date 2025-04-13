import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

@st.cache_resource()
def load_data(path):
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(path, header=None, names=column_names, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    return df

@st.cache_resource()
def preprocess(df):
    X = df.drop("income", axis=1)
    y = df["income"]

    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    label_encoders["income"] = y_le  # include target if needed

    return X, y, label_encoders

def decode_features(df, encoders):
    df_decoded = df.copy()
    for col, le in encoders.items():
        if col in df_decoded.columns:
            df_decoded[col] = le.inverse_transform(df_decoded[col])
    return df_decoded

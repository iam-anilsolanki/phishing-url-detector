import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Phishing URL Detector", layout="wide")
st.title("🔐 Phishing URL Detection App")

# -----------------------
# Extract Features from URL
# -----------------------
def extract_features_from_url(url):
    parsed = urlparse(url)
    features = {}

    features["NumDots"] = url.count(".")
    features["UrlLength"] = len(url)
    features["NumDash"] = url.count("-")
    features["NumQueryComponents"] = len(parsed.query.split("&")) if parsed.query else 0
    features["PathLevel"] = len([p for p in parsed.path.split("/") if p])
    features["PathLength"] = len(parsed.path)
    features["NumNumericChars"] = len(re.findall(r'\d', url))

    return pd.DataFrame([features])

# -----------------------
# Train Model Internally
# -----------------------
@st.cache_data
def train_model():
    df = pd.read_csv("Phising_dataset_predict.csv")
    df = df.dropna(subset=["Phising"])
    df = df.dropna()

    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)

    # Features and target
    X = df.drop("Phising", axis=1)
    y = df["Phising"]

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_names = train_model()

# -----------------------
# Streamlit Input
# -----------------------
st.subheader("🔗 Enter a URL to analyze:")
user_url = st.text_input("Paste a URL here", placeholder="https://example.com/login")

# -----------------------
# Predict from URL
# -----------------------
if st.button("🧠 Predict"):
    if user_url:
        features_df = extract_features_from_url(user_url)

        # Ensure column order matches training set
        features_df = features_df[feature_names]

        prediction = model.predict(features_df)[0]
        result = "🟢 Legitimate Website" if prediction == 0 else "🔴 Phishing Website"

        st.subheader("🔎 Prediction Result:")
        st.success(result)
    else:
        st.warning("Please enter a valid URL.")

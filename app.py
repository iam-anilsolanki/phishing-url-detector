import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Phishing URL Detector", layout="wide")
st.title("🔐 Phishing URL Detection App")

# -----------------------
# Step 1: Train Model In-App
# -----------------------
@st.cache_data
def train_model():
    # Load the dataset
    df = pd.read_csv("Phising_dataset_predict.csv")

    # Clean the dataset
    df = df.dropna(subset=["Phising"])  # Drop rows where label is missing
    df = df.dropna()  # Drop rows with any missing values

    # Drop constant columns (like AtSymbol, etc., if needed)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)

    # Split features and target
    X = df.drop("Phising", axis=1)
    y = df["Phising"]

    # Train model
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Return model and list of feature names
    return model, X.columns.tolist()

model, feature_names = train_model()

# -----------------------
# Step 2: Collect Inputs
# -----------------------
st.subheader("🔍 Manually Enter URL Features:")

input_values = {}
for feature in feature_names:
    input_values[feature] = st.number_input(f"{feature}", min_value=0, step=1)

# -----------------------
# Step 3: Predict
# -----------------------
if st.button("🧠 Predict"):
    input_array = np.array([list(input_values.values())])
    prediction = model.predict(input_array)[0]

    result = "🟢 Legitimate Website" if prediction == 0 else "🔴 Phishing Website"
    st.subheader("🔎 Prediction Result:")
    st.success(result)

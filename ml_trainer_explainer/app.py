import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import os
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Ensure necessary directories exist on Streamlit Cloud
if not os.path.exists("models"):
    os.makedirs("models")

# Function to save and load models (Handles Streamlit Cloud Persistence)
def save_model(model, feature_names, label_encoders, filename):
    with open(filename, 'wb') as f:
        pickle.dump((model, feature_names, label_encoders), f)

def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        st.warning("⚠️ Model file not found! Please train a model first.")
        return None

# Streamlit UI setup
st.set_page_config(page_title="ML Trainer & Explainer", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Train Models", "📂 Upload Data & Predict", "📈 Visualization & Explainability"], index=0)

# Home Page
if page == "🏠 Home":
    st.title("Welcome to ML Trainer & Explainer 🚀")
    st.markdown("""
        ## 🔥 Features
        - 🏋️ Train ML models (Logistic Regression, Random Forest, SVM, XGBoost)
        - 💾 Save & load trained models
        - 📂 Upload datasets & make predictions
        - 📊 Analyze model explainability using SHAP & Evidently AI
    """)

# Train Models Page
elif page == "📊 Train Models":
    st.title("Train Machine Learning Models 🎯")

    uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])
    df = None  # Initialize df variable

    # If no file uploaded, use a hosted default dataset
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        default_csv_path = "https://raw.githubusercontent.com/yourusername/yourrepo/main/sample.csv"
        df = pd.read_csv(default_csv_path)
        st.info("ℹ️ No file uploaded. Using a sample dataset from GitHub.")

    if df is not None:
        st.write("### 📌 Dataset Preview", df.head())
        target = st.selectbox("🎯 Select target column", df.columns)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        }

        selected_model = st.selectbox("🤖 Choose Model", list(models.keys()))

        if st.button("🚀 Train & Save Model"):
            X = df.drop(columns=[target])
            y = df[target]

            # Encode categorical features
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            model = models[selected_model]
            model.fit(X_train, y_train)

            filename = f"{selected_model.replace(' ', '_')}.pkl"
            save_model(model, list(X.columns), label_encoders, filename)
            st.success(f"✅ {selected_model} trained and saved as {filename}")

# Upload Data & Predict Page
elif page == "📂 Upload Data & Predict":
    st.title("Upload Data & Make Predictions 📌")
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]

    if model_files:
        selected_model_file = st.selectbox("📂 Select a trained model", model_files)
    else:
        st.error("⚠️ No trained models found! Please train a model first.")
        st.stop()

    uploaded_file = st.file_uploader("📤 Upload a CSV file for prediction", type=["csv"])
    df = None  # Initialize df variable

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        default_csv_path = "https://raw.githubusercontent.com/yourusername/yourrepo/main/sample.csv"
        df = pd.read_csv(default_csv_path)
        st.info("ℹ️ No file uploaded. Using a sample dataset from GitHub.")

    if selected_model_file and df is not None:
        model, feature_names, label_encoders = load_model(selected_model_file)
        st.write("### 🔍 Selected Data Preview", df.head())

        if model:
            X = df[feature_names]

            # Apply label encoders if needed
            for col, le in label_encoders.items():
                if col in X.columns:
                    X[col] = le.transform(X[col])

            predictions = model.predict(X)
            st.write("### 📌 Predictions:", predictions)

# Visualization & Explainability Page
elif page == "📈 Visualization & Explainability":
    st.title("Model Explainability & Data Visualization 📊")
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]

    if model_files:
        selected_model_file = st.selectbox("📂 Select a trained model", model_files)
    else:
        st.error("⚠️ No trained models found! Please train a model first.")
        st.stop()

    uploaded_file = st.file_uploader("📤 Upload a CSV file for visualization", type=["csv"])
    df = None  # Initialize df variable

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        default_csv_path = "https://raw.githubusercontent.com/yourusername/yourrepo/main/sample.csv"
        df = pd.read_csv(default_csv_path)
        st.info("ℹ️ No file uploaded. Using a sample dataset from GitHub.")

    if selected_model_file and df is not None:
        model, feature_names, label_encoders = load_model(selected_model_file)

        if st.button("📊 Generate Evidently Data Drift Report"):
            reference_data = df.sample(frac=0.5, random_state=42)
            current_data = df.drop(reference_data.index)

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=reference_data, current_data=current_data)

            # Convert report into an HTML string
            html_content = drift_report.get_html()
            st.components.v1.html(html_content, height=800, scrolling=True)

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

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Function to save model along with feature names and label encoders
def save_model(model, feature_names, label_encoders, filename):
    with open(os.path.join("models", filename), 'wb') as f:
        pickle.dump((model, feature_names, label_encoders), f)

# Function to load model along with feature names and label encoders
def load_model(filename):
    with open(os.path.join("models", filename), 'rb') as f:
        return pickle.load(f)  # Returns (model, feature_names, label_encoders)

# Streamlit UI configuration
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

    data_option = st.radio("Select Data Source", ["Upload File", "Select from Data Folder"], key="data_source")
    df = None  # Initialize df variable

    if data_option == "Upload File":
        uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"], key="train_upload")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    else:
        csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
        if csv_files:
            selected_csv = st.selectbox("📂 Select a dataset", csv_files, key="data_select")
            if selected_csv:
                df = pd.read_csv(os.path.join("data", selected_csv))

    if df is not None:
        st.write("### 📌 Dataset Preview", df.head())
        target = st.selectbox("🎯 Select target column", df.columns, key="target_col")

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        }

        selected_model = st.selectbox("🤖 Choose Model", list(models.keys()), key="model_select")

        if st.button("🚀 Train & Save Model", key="train_button"):
            X = df.drop(columns=[target])
            y = df[target]

            # Encode categorical features
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le  # Store encoder for later use

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
        selected_model_file = st.selectbox("📂 Select a trained model", model_files, key="model_file_select")
    else:
        st.error("⚠️ No trained models found! Please train a model first.")
        st.stop()

    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    if csv_files:
        selected_csv = st.selectbox("📂 Select a dataset for prediction", csv_files, key="data_select")
    else:
        st.error("⚠️ No datasets found! Please upload a dataset first.")
        st.stop()

    if selected_model_file and selected_csv:
        model, feature_names, label_encoders = load_model(selected_model_file)
        data = pd.read_csv(os.path.join("data", selected_csv))
        st.write("### 🔍 Selected Data Preview", data.head())

# Visualization & Explainability Page
elif page == "📈 Visualization & Explainability":
    st.title("Model Explainability & Data Visualization 📊")
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]

    if model_files:
        selected_model_file = st.selectbox("📂 Select a trained model", model_files, key="viz_model_select")
    else:
        st.error("⚠️ No trained models found! Please train a model first.")
        st.stop()

    if csv_files:
        selected_csv = st.selectbox("📂 Select a dataset for visualization", csv_files, key="viz_data_select")
    else:
        st.error("⚠️ No datasets found! Please upload a dataset first.")
        st.stop()

    if selected_model_file and selected_csv:
        model, feature_names, label_encoders = load_model(selected_model_file)
        data = pd.read_csv(os.path.join("data", selected_csv))

        if st.button("📊 Generate Evidently Data Drift Report", key="evidently_button"):
            reference_data = data.sample(frac=0.5, random_state=42)
            current_data = data.drop(reference_data.index)
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=reference_data, current_data=current_data)
            drift_report.save_html("evidently_report.html")

            with open("evidently_report.html", "r", encoding="utf-8") as file:
                html_content = file.read()
            st.components.v1.html(html_content, height=800, scrolling=True)

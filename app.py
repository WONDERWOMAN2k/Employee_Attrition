import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Load model safely
@st.cache_data
def load_model():
    model_path = "attrition_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("🚨 Model file 'attrition_model.pkl' not found! Please upload it.")
        return None

# Load cleaned data safely
@st.cache_data
def load_data():
    data_path = "cleaned_employee_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("🚨 Data file 'cleaned_employee_data.csv' not found! Please upload it.")
        return pd.DataFrame()

# Load model and data
model = load_model()
full_df = load_data()

# Fallback for missing EmployeeID
if not full_df.empty and "EmployeeID" not in full_df.columns:
    full_df["EmployeeID"] = range(1, len(full_df) + 1)

# Title and description
st.title("🏢 Employee Attrition Prediction Dashboard")
st.markdown("Analyze employee attrition, job satisfaction, and performance with real-time predictions and visual insights.")

# Sidebar menu
menu = st.sidebar.selectbox("Choose Feature", (
    "Attrition Prediction", 
    "High-Risk Employee List", 
    "Job Satisfaction & Performance", 
    "Side-by-Side Comparison")
)

# 1. Attrition Prediction
if menu == "Attrition Prediction":
    st.header("📊 Real-Time Attrition Prediction")
    uploaded_file = st.file_uploader("Upload employee data (CSV)", type="csv")

    if uploaded_file and model:
        input_df = pd.read_csv(uploaded_file)

        # Ensure consistent preprocessing
        if 'Attrition' in input_df.columns:
            input_df = input_df.drop(columns=['Attrition'])

        # Align features with the model
        model_feature_names = model.feature_names_in_
        input_df = input_df.reindex(columns=model_feature_names, fill_value=0)

        try:
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)

            input_df["Attrition Prediction"] = prediction
            input_df["Risk Score (%)"] = (proba[:, 1] * 100).round(2)

            st.success("✅ Prediction complete!")
            st.dataframe(input_df)

            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "attrition_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# 2. High-Risk Employee List
elif menu == "High-Risk Employee List":
    st.header("🚨 High-Risk Employee List")
    st.markdown("Employees with a risk score greater than 70% are considered high risk.")

    required_cols = {"EmployeeID", "Department", "JobRole"}
    if not full_df.empty and model:
        if required_cols.issubset(full_df.columns):
            if "Risk Score (%)" not in full_df.columns:
                try:
                    proba = model.predict_proba(full_df)
                    full_df["Risk Score (%)"] = (proba[:, 1] * 100).round(2)
                except Exception as e:
                    st.error(f"⚠️ Error calculating risk score: {e}")

            high_risk = full_df[full_df["Risk Score (%)"] >= 70]
            st.dataframe(high_risk[["EmployeeID", "Department", "JobRole", "Risk Score (%)"]])
        else:
            st.warning("⚠️ Required columns missing for risk analysis.")

# 3. Job Satisfaction & Performance
elif menu == "Job Satisfaction & Performance":
    st.header("⭐ Job Satisfaction & Performance Analysis")

    required_columns = ['JobSatisfaction', 'PerformanceRating']
    if all(col in full_df.columns for col in required_columns):
        st.subheader("Heatmap: Satisfaction vs Performance")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(pd.crosstab(full_df["JobSatisfaction"], full_df["PerformanceRating"]),
                    annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

        st.subheader("High Satisfaction + High Performance Employees")
        top_employees = full_df[
            (full_df["JobSatisfaction"] >= 4) & 
            (full_df["PerformanceRating"] >= 4)
        ]
        st.dataframe(top_employees[["EmployeeID", "JobRole", "JobSatisfaction", "PerformanceRating"]])
    else:
        st.warning(f"⚠️ Missing one or more required columns: {required_columns}")

# 4. Side-by-Side Comparison
elif menu == "Side-by-Side Comparison":
    st.header("🧍‍♂️🧍 Employee Comparison")

    if not full_df.empty:
        if "EmployeeID" in full_df.columns:
            employee_ids = full_df["EmployeeID"].unique()
            emp1 = st.selectbox("Select Employee 1", employee_ids)

            if len(employee_ids) > 1:
                emp2 = st.selectbox("Select Employee 2", employee_ids, index=1)
            else:
                emp2 = st.selectbox("Select Employee 2", employee_ids)

            emp1_data = full_df[full_df["EmployeeID"] == emp1].T
            emp2_data = full_df[full_df["EmployeeID"] == emp2].T

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"👤 Employee {emp1}")
                st.dataframe(emp1_data)

            with col2:
                st.subheader(f"👤 Employee {emp2}")
                st.dataframe(emp2_data)
        else:
            st.warning("⚠️ 'EmployeeID' column not found for comparison.")

# Footer
st.markdown("---")
st.markdown("© 2025 HR Analytics Project | Built with ❤️ using Streamlit")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Load model and data
@st.cache_data
def load_model():
    return joblib.load("attrition_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_employee_data.csv")

model = load_model()
full_df = load_data()

# Title
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

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        input_df["Attrition Prediction"] = prediction
        input_df["Risk Score (%)"] = (proba[:, 1] * 100).round(2)

        st.success("✅ Prediction complete!")
        st.dataframe(input_df)

        # Download
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "attrition_predictions.csv", "text/csv")

# 2. High-Risk Employee List
elif menu == "High-Risk Employee List":
    st.header("🚨 High-Risk Employee List")
    st.markdown("Employees with risk score greater than 70% are considered high risk.")

    if "Risk Score (%)" not in full_df.columns:
        proba = model.predict_proba(full_df)
        full_df["Risk Score (%)"] = (proba[:, 1] * 100).round(2)

    high_risk = full_df[full_df["Risk Score (%)"] >= 70]
    st.dataframe(high_risk[["EmployeeID", "Department", "JobRole", "Risk Score (%)"]])

# 3. Job Satisfaction & Performance
elif menu == "Job Satisfaction & Performance":
    st.header("⭐ Job Satisfaction & Performance Analysis")

    st.subheader("Heatmap: Satisfaction vs Performance")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(pd.crosstab(full_df["JobSatisfaction"], full_df["PerformanceRating"]), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("High Satisfaction + High Performance Employees")
    top_employees = full_df[(full_df["JobSatisfaction"] >= 4) & (full_df["PerformanceRating"] >= 4)]
    st.dataframe(top_employees[["EmployeeID", "JobRole", "JobSatisfaction", "PerformanceRating"]])

# 4. Side-by-Side Comparison
elif menu == "Side-by-Side Comparison":
    st.header("🧍‍♂️🧍 Employee Comparison")

    employee_ids = full_df["EmployeeID"].unique()
    emp1 = st.selectbox("Select Employee 1", employee_ids)
    emp2 = st.selectbox("Select Employee 2", employee_ids, index=1)

    emp1_data = full_df[full_df["EmployeeID"] == emp1].T
    emp2_data = full_df[full_df["EmployeeID"] == emp2].T

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"👤 Employee {emp1}")
        st.dataframe(emp1_data)

    with col2:
        st.subheader(f"👤 Employee {emp2}")
        st.dataframe(emp2_data)

# Footer
st.markdown("---")
st.markdown("© 2025 HR Analytics Project | Built with ❤️ using Streamlit")

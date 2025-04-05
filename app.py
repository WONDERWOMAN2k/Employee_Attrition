import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# Title
st.title("👩‍💼 Employee Attrition Prediction App")

# Load model
@st.cache_resource
def load_model():
    model_path = "attrition_model.pkl"
    return joblib.load(model_path)

model = load_model()

# Upload data
uploaded_file = st.file_uploader("📂 Upload Cleaned Employee Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    # Ensure model features exist
    model_features = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
        'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
        'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]
    
    missing_features = [col for col in model_features if col not in df.columns]
    if missing_features:
        st.error(f"❌ Missing required features: {missing_features}")
        st.stop()

    # Predict attrition
    X = df[model_features]
    prediction = model.predict(X)
    df['Attrition_Prediction'] = prediction

    st.subheader("🎯 Prediction Results")
    st.dataframe(df[['Attrition_Prediction']])

    # Risk analysis
    st.subheader("⚠️ Risk Analysis")
    required_columns = ['JobSatisfaction', 'PerformanceRating']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.warning(f"⚠️ Missing one or more required columns for risk analysis: {missing_cols}")
    else:
        st.write("📊 Risk Summary:")
        risk_summary = df.groupby(['JobSatisfaction', 'PerformanceRating'])['Attrition_Prediction'].value_counts().unstack().fillna(0)
        st.dataframe(risk_summary)

        st.write("📉 Heatmap of Attrition Risk")
        pivot_table = df.pivot_table(index='JobSatisfaction', columns='PerformanceRating', values='Attrition_Prediction', aggfunc=np.mean)
        fig, ax = plt.subplots()
        sns.heatmap(pivot_table, annot=True, cmap="Reds", fmt=".2f", ax=ax)
        st.pyplot(fig)

else:
    st.info("Please upload the `cleaned_employee_data.csv` file to proceed.")

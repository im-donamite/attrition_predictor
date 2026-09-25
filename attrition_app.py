import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the NEW Logistic Regression model
# This model was trained using our selected 12 features.
model = joblib.load("attrition_model_new.pkl")


# Load the scaler used during training.
# New employee data must be scaled using this same scaler.
scaler = joblib.load("attrition_scaler_new.pkl")


# Load the feature columns created after one-hot encoding.
# This ensures the app gives the model the exact columns it expects.
feature_columns = joblib.load("feature_columns_new.pkl")

# Page configuration
st.set_page_config(
    page_title="Attrition - Employee Attrition Predictor",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
# This gives the app a pink and purple theme instead of
# Streamlit's default appearance.

st.markdown("""
    <style>

        .title {
            text-align: center;
            font-size: 38px;
            color: #9D4EDD;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #6F6875;
            font-size: 18px;
            margin-bottom: 25px;
        }

        .stButton>button {
            width: 100%;
            background-color: #FF8FAB;
            color: white;
            font-size: 18px;
            font-weight: 600;
            padding: 10px;
            border-radius: 15px;
            border: none;
        }

        .stButton>button:hover {
            background-color: #9D4EDD;
            color: white;
        }

    </style>
""", unsafe_allow_html=True)

# Main heading
st.markdown(
    '<div class="title">💼 AttritionPredictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Employee Attrition Predictor</div>',
    unsafe_allow_html=True
)



st.markdown("### 👤 Enter Employee Details")


col1, col2 = st.columns(2)



with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=30,
        step=1
    )

    business_travel = st.selectbox(
        "Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )

    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4],
        index=2
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5],
        index=1
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4],
        index=2
    )



with col2:

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=100,
        max_value=25000,
        value=5000,
        step=100
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=5,
        step=1
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5,
        step=1
    )


input_data = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [business_travel],
    "DistanceFromHome": [distance_from_home],
    "JobInvolvement": [job_involvement],
    "JobLevel": [job_level],
    "JobRole": [job_role],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status],
    "MonthlyIncome": [monthly_income],
    "OverTime": [overtime],
    "TotalWorkingYears": [total_working_years],
    "YearsAtCompany": [years_at_company]
})

#next convert categorical values
input_data = pd.get_dummies(
    input_data,
    columns=[
        "BusinessTravel",
        "JobRole",
        "MaritalStatus",
        "OverTime"
    ],
    drop_first=True
)

input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

input_scaled = scaler.transform(input_data)



# Create a button that runs the prediction when clicked.
if st.button("🔍 Predict Employee Attrition"):

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    attrition_probability = probabilities[1]
    st.write("Prediction:", prediction)
    st.write("Probability of Stay:", probabilities[0])
    st.write("Probability of Leave:", probabilities[1])
    if prediction == 1:
        result = "Likely to Leave"
    else:
        result = "Likely to Stay"

    # Display the result.
    st.markdown("### Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ {result}")
    else:
        st.success(f"✅ {result}")

    # Display probability
    st.write(
        f"Probability of Attrition: "
        f"{attrition_probability * 100:.2f}%"
    )

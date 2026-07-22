import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_disease_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page title
st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details to predict the possibility of heart disease.")

st.divider()

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

dataset = st.selectbox(
    "Dataset",
    ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
)

cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal pain",
        "asymptomatic"
    ]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50.0,
    max_value=250.0,
    value=120.0
)

chol = st.number_input(
    "Cholesterol",
    min_value=0.0,
    max_value=700.0,
    value=200.0
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [False, True]
)

restecg = st.selectbox(
    "Resting ECG",
    [
        "normal",
        "ST-T wave abnormality",
        "left ventricular hypertrophy"
    ]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50.0,
    max_value=250.0,
    value=150.0
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [False, True]
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    [
        "upsloping",
        "flat",
        "downsloping"
    ]
)

if st.button("Predict Heart Disease"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "dataset": [dataset],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [int(fbs)],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [int(exang)],
        "oldpeak": [oldpeak],
        "slope": [slope]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["sex", "dataset", "cp", "restecg", "slope"]
    )

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Heart Disease Detected")
    else:
        st.success("✅ Prediction: No Heart Disease Detected")
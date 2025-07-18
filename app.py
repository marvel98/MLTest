import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('heart_failure_model.joblib')

st.title("Heart Failure Prediction App")

st.write("""
This app predicts the **likelihood of death** during the follow-up period for heart failure patients.
""")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 60)
    anaemia = st.sidebar.selectbox('Anaemia', [0, 1])
    creatinine_phosphokinase = st.sidebar.slider('CPK Level', 0, 8000, 250)
    diabetes = st.sidebar.selectbox('Diabetes', [0, 1])
    ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 10, 80, 38)
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', [0, 1])
    platelets = st.sidebar.slider('Platelets', 10000, 900000, 250000)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.1, 10.0, 1.0)
    serum_sodium = st.sidebar.slider('Serum Sodium', 110, 150, 137)
    sex = st.sidebar.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    smoking = st.sidebar.selectbox('Smoking', [0, 1])
    time = st.sidebar.slider('Follow-up Time (days)', 1, 300, 100)

    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user input
st.subheader("Patient Data")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

# Display result
st.subheader("Prediction")
st.write("🔴 Death Event Likely" if prediction == 1 else "🟢 Survival Likely")

st.subheader("Prediction Probability")
st.write(f"Probability of Death Event: {prediction_proba:.2%}")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and dataset
MODEL_PATH = "heart_failure_model.joblib"
DATA_PATH = "heart_failure_clinical_records_dataset.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

st.title("💓 Heart Failure Prediction App")

st.markdown("""
This app predicts the **likelihood of death** during follow-up in heart failure patients based on clinical records.
""")

# --- Section: Dataset Overview ---
st.subheader("📊 Dataset Overview")
st.write(df.head())

with st.expander("View Data Summary"):
    st.write(df.describe())
    st.write("Class Balance:")
    st.bar_chart(df["DEATH_EVENT"].value_counts())

# --- Section: Patient Input ---
st.sidebar.header("🧑‍⚕️ Enter Patient Data")

def get_user_input():
    age = st.sidebar.slider("Age", 20, 100, 60)
    anaemia = st.sidebar.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
    creatinine_phosphokinase = st.sidebar.slider("CPK Level", 20, 8000, 250)
    diabetes = st.sidebar.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
    ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 10, 80, 38)
    high_blood_pressure = st.sidebar.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
    platelets = st.sidebar.slider("Platelets", 10000, 900000, 250000)
    serum_creatinine = st.sidebar.slider("Serum Creatinine", 0.1, 10.0, 1.0)
    serum_sodium = st.sidebar.slider("Serum Sodium", 110, 150, 137)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    smoking = st.sidebar.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
    time = st.sidebar.slider("Follow-up Time (days)", 1, 300, 100)

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

input_df = get_user_input()

st.subheader("📝 Patient Data")
st.write(input_df)

# --- Section: Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔮 Prediction")
    if prediction == 1:
        st.error(f"⚠️ Likely Death Event\nProbability: {proba:.2%}")
    else:
        st.success(f"✅ Likely Survival\nProbability of Death Event: {proba:.2%}")

# --- Section: Feature Importance ---
st.subheader("📌 Feature Importance")

importances = model.feature_importances_
features = df.drop("DEATH_EVENT", axis=1).columns

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=features, ax=ax)
ax.set_title("Feature Importances from Random Forest")
st.pyplot(fig)

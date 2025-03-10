import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title("🚢 Titanic Survival Estimator")

# Load trained model
survival_predictor = pickle.load(open('titanic_logistic.pkl', 'rb'))

# User input fields
class_ticket = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
gender_passenger = st.selectbox("Sex", ["Male", "Female"])
age_value = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp_count = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch_count = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
fare_amount = st.number_input("Fare", min_value=0.0, step=0.1)
embark_location = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert categorical inputs to match model's training format
male_flag = 1 if gender_passenger == "Male" else 0  # Model was trained with 'Sex_male'
embarked_Q_flag = 1 if embark_location == "Q" else 0
embarked_S_flag = 1 if embark_location == "S" else 0

# Organize user input in the correct format
user_data = {
    'Pclass': class_ticket,
    'Age': age_value,
    'SibSp': sibsp_count,
    'Parch': parch_count,
    'Fare': fare_amount,
    'Sex_male': male_flag,
    'Embarked_Q': embarked_Q_flag,
    'Embarked_S': embarked_S_flag
}

# Convert to DataFrame with correct structure
input_df = pd.DataFrame(user_data, index=[0])

# Predict button
if st.button("Check Survival"):
    survival_result = survival_predictor.predict(input_df)

    # Display prediction outcome
    survival_message = "Survived 🟢" if survival_result[0] == 1 else "Did Not Survive 🔴"
    st.success(f"Prediction: *{survival_message}*")

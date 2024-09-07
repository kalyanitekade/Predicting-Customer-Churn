import streamlit as st
import joblib
import numpy as np

# Load the pre-trained Random Forest model
model = joblib.load('D:/Project Pro/Predicting Customer Churn/output/models/model_Random.pkl')


# Define the UI with only the top 5 features
st.title("Customer Churn Prediction")

# Collect user inputs for the top 5 features
customer_support_calls = st.number_input("Customer Support Calls", min_value=0, max_value=50, value=2)
maximum_days_inactive = st.number_input("Maximum Days Inactive", min_value=0, max_value=365, value=10)
maximum_daily_mins = st.number_input("Maximum Daily Minutes", min_value=0, max_value=1440, value=120)
weekly_mins_watched = st.number_input("Weekly Minutes Watched", min_value=0, max_value=10000, value=500)
videos_watched = st.number_input("Videos Watched", min_value=0, max_value=1000, value=100)

# Default values for the other 4 features that were part of the original model
default_age = 35  # Example default value for age
default_no_of_days_subscribed = 200  # Example default value for days subscribed
default_weekly_max_night_mins = 100  # Example default value for night minutes
default_minimum_daily_mins = 10  # Example default value for minimum daily minutes

# Collect the user input into an array for prediction, including default values
input_data = np.array([[
    customer_support_calls,
    maximum_days_inactive,
    maximum_daily_mins,
    weekly_mins_watched,
    videos_watched,
    default_age,
    default_no_of_days_subscribed,
    default_weekly_max_night_mins,
    default_minimum_daily_mins
]])

# Predict churn probability using the model
if st.button("Predict Churn"):
    prediction = model.predict_proba(input_data)[0][1]  # Get the probability of "Churn"
    
    if prediction >= 0.5:
        st.warning(f"The customer is likely to churn with a probability of {prediction:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {1 - prediction:.2f}")


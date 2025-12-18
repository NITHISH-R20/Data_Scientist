import streamlit as st
import numpy as np
import pickle

# Load saved model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 Student Performance Prediction")
st.write("Predict final score using Linear Regression")

st.divider()

# Input fields (adjust count/order to match your model)
feature1 = st.number_input("Study Hours", min_value=0.0)
feature2 = st.number_input("Attendance (%)", min_value=0.0)
feature3 = st.number_input("Previous Score", min_value=0.0)
feature4 = st.number_input("Assignments Score", min_value=0.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")

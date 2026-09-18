import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

# Load the trained model
model_path = Path(__file__).parent / "electricity_bill_model.pkl"
model = joblib.load(model_path)

# Create polynomial transformer
poly = PolynomialFeatures(degree=2)

# App title
st.title("Electricity Bill Predictor")

st.write(
    "Enter the AC Units to predict the expected Electric Bill."
)

# User input
ac_units = st.number_input(
    "AC Units",
    min_value=0.0,
    step=5.0
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    # Transform input using PolynomialFeatures
    input_data_poly = poly.fit_transform(input_data)

    # Predict Electric Bill
    prediction = model.predict(input_data_poly)[0]

    if prediction >= 0:
        st.success(
            f"Expected Electric Bill: ₹{prediction:.2f}"
        )
    else:
        st.error("Error Occurred")

import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_sales_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Sales Pitch Prediction App")
st.write("""
This application predicts the likelihood of a sales pitch success based on its operational parameters.
Enter the  data below to get a prediction.
""")

Age         = st.number_input("Age", 1, 100, 1)
CityTier    = st.selectbox("CityTier", [1,2,3])
DurationOfPitch = st.number_input("DurationOfPitch", ["Free Lancer", "Salaried", "Salaried","Large Business"])
Occupation   = st.selectbox("Rotational Speed (RPM)", 0, 3000, 1400)
Gender       = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting    = st.number_input("NumberOfPersonVisiting", 0, 100, 1)
NumberOfFollowups  = st.number_input("NumberOfFollowups", 0, 100, 1)
ProductPitched = st.number_input("ProductPitched", ["Deluxe", "Basic", "Standard","King","Super Deluxe"])
MaritalStatus = st.number_input("MaritalStatus", ["Married", "Unmarried", "Single","Divorced",])
NumberOfTrips = st.number_input("NumberOfTrips", 0, 100, 1)
Passport = st.number_input("Passport", 1,0)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", 0, 5, 1)
OwnCar = st.number_input("OwnCar", 1,0)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", 0, 5, 1)
Designation = st.selectbox("Designation", ["Executive", "AVP","Senior Manager", "Manager","VP"])
MonthlyIncome = st.number_input("MonthlyIncome", 1, 100000, 1)






input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier ": CityTier ,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Gender": Gender,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("ProdTaken"):
    prediction = model.predict(input_data)[0]
    result = "ProdTaken" if prediction == 1 else "ProdNotTaken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

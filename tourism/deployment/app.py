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

Age = st.number_input("Age", min_value=18, max_value=90, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("CityTier", [1, 2, 3], index=0)
DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=120, value=10)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business", "Government Sector"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=10, value=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=1)
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "King", "Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred Property Star (1-5)", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Unmarried", "Single", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=50, value=1)
Passport = st.selectbox("Passport (0=No, 1=Yes)", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (0-5)", min_value=0, max_value=5, value=3)
OwnCar = st.selectbox("Own Car (0=No, 1=Yes)", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Executive", "AVP", "Senior Manager", "Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=25000)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("Predict ProdTaken"): # Changed button text for clarity
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

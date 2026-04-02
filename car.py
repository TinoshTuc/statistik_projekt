import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import joblib
import streamlit as st

## Code for streamlit
## Load the modell
model = joblib.load("car_price_model.pkl")

st.title("Car Price Predictor")
st.write("Enter car details to estimate the price:")

# Numerical inputs
year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
engine_size = st.number_input("Engine Size", value=2.0)
mileage = st.number_input("Mileage", value=50000)
doors = st.number_input("Doors", value=4)
owner_count = st.number_input("Owner Count", value=1)

# Categorical inputs
brand = st.selectbox("Brand", [
    "BMW","Chevrolet","Ford","Honda","Hyundai","Kia","Mercedes","Toyota","Volkswagen"
])

model_name = st.selectbox("Model", [
    "5 Series","A3","A4","Accord","C-Class","CR-V","Camry","Civic",
    "Corolla","E-Class","Elantra","Equinox","Explorer","Fiesta","Focus",
    "GLA","Golf","Impala","Malibu","Optima","Passat","Q5","RAV4",
    "Rio","Sonata","Sportage","Tiguan","Tucson","X5"
])

fuel = st.selectbox("Fuel Type", ["Electric","Hybrid","Petrol"])
transmission = st.selectbox("Transmission", ["Manual","Semi-Automatic"])

# Create input dataframe (ALL columns!)
input_data = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

# Numerical values
input_data.at[0, "Year"] = year
input_data.at[0, "Engine_Size"] = engine_size
input_data.at[0, "Mileage"] = mileage
input_data.at[0, "Doors"] = doors
input_data.at[0, "Owner_Count"] = owner_count

# Dummy variables
input_data.at[0, f"Brand_{brand}"] = 1
input_data.at[0, f"Model_{model_name}"] = 1
input_data.at[0, f"Fuel_Type_{fuel}"] = 1
input_data.at[0, f"Transmission_{transmission}"] = 1

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")
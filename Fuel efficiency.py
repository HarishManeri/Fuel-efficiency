import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Define functions to map categorical data
def map_vehicle_type(vehicle_type):
    mapping = {'bike': 1, 'car': 2, 'bus': 3, 'truck': 4, 'scooter': 5}
    return mapping.get(vehicle_type, 0)

def map_fuel_type(fuel_type):
    mapping = {'petrol': 1, 'diesel': 2, 'electric': 3, 'CNG': 4}
    return mapping.get(fuel_type, 0)

def predict_fuel_efficiency(vehicle_type, fuel_type, weight, displacement, horsepower, speed):
    # Placeholder training data
    sample_data = {
        'vehicle_type': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'fuel_type': [1, 1, 2, 2, 3, 1, 2, 1, 2, 3],
        'weight': [100, 1200, 5000, 10000, 150, 120, 1500, 7000, 12000, 200],
        'displacement': [150, 1500, 6000, 10000, 110, 180, 2000, 8000, 11000, 125],
        'horsepower': [15, 100, 250, 400, 8, 18, 150, 300, 450, 10],
        'speed': [40, 60, 50, 40, 45, 50, 70, 60, 50, 55],
        'fuel_efficiency': [60, 20, 10, 5, 65, 50, 15, 8, 4, 70]
    }
    train_df = pd.DataFrame(sample_data)
    X = train_df.drop('fuel_efficiency', axis=1)
    y = train_df['fuel_efficiency']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    input_data = pd.DataFrame({
        'vehicle_type': [map_vehicle_type(vehicle_type)],
        'fuel_type': [map_fuel_type(fuel_type)],
        'weight': [weight],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'speed': [speed]
    })
    
    predicted_efficiency = model.predict(input_data)
    return predicted_efficiency[0]

st.title("Fuel Efficiency Prediction")

vehicle_type = st.selectbox("Select vehicle type", ["bike", "car", "bus", "truck", "scooter"])
fuel_type = st.selectbox("Select fuel type", ["petrol", "diesel", "electric", "CNG"])
weight = st.number_input("Enter vehicle weight (in kg)", min_value=50.0)
displacement = st.number_input("Enter engine displacement (in cc)", min_value=50.0)
horsepower = st.number_input("Enter horsepower", min_value=1.0)
speed = st.number_input("Enter average speed (in km/h)", min_value=10.0)

if st.button("Predict Fuel Efficiency"):
    efficiency = predict_fuel_efficiency(vehicle_type, fuel_type, weight, displacement, horsepower, speed)
    st.success(f"Predicted fuel efficiency: {efficiency:.2f} km/l")

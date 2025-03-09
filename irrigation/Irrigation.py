import time
import random

# Constants
FIELD_CAPACITY = 30
WILTING_POINT = 10
TRIGGER_THRESHOLD = 15

# Dictionary of crop water requirements (mm/day)
crop_water_requirements = {
    'apple': 5.5,
    'banana': 6.0,
    'blackgram': 4.0,
    'chickpea': 3.5,
    'coconut': 8.0,
    'coffee': 6.5,
    'cotton': 7.0,
    'grapes': 4.5,
    'jute': 5.0,
    'kidneybeans': 4.0,
    'lentil': 3.0,
    'maize': 6.0,
    'mango': 5.5,
    'mothbeans': 4.2,
    'mungbean': 3.8,
    'muskmelon': 6.2,
    'orange': 5.0,
    'papaya': 6.5,
    'pigeonpeas': 4.0,
    'pomegranate': 4.3,
    'rice': 8,
    'watermelon': 6.0,
}


# Function to calculate water needed based on current soil moisture and crop's water requirement
def calculate_irrigation(soil_moisture, crop_name):
    crop_water_requirement = crop_water_requirements.get(crop_name.lower())
    deficit = FIELD_CAPACITY - soil_moisture - crop_water_requirement
    water_needed = max(0, deficit)  # Ensure no negative values
    return water_needed


# Function to simulate sensor reading (replace with real sensor data)
def get_soil_moisture():
    return random.uniform(0, 30)


# Function to send irrigation signal
def send_irrigation_signal(water_needed):
    if water_needed > 0:
        print(f"Irrigation needed: {water_needed:.2f} mm of water")
        print("Signal sent to irrigation system to begin watering.")
    else:
        print("No irrigation needed.")


# Main monitoring loop
def monitor_irrigation(crop_name):
    while True:
        soil_moisture = get_soil_moisture()
        print(f"Current soil moisture: {soil_moisture:.2f} mm")

        if soil_moisture < TRIGGER_THRESHOLD:
            water_needed = calculate_irrigation(soil_moisture, crop_name)
            send_irrigation_signal(water_needed)x
        else:
            print("Soil moisture is sufficient. No irrigation required.")

        # Check at intervals
        time.sleep(20)



crop_name = 'rice'  # Example crop
monitor_irrigation(crop_name)

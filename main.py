import pandas as pd
import time
import requests
import os
import pyttsx3

engine = pyttsx3.init()

base_dir = os.path.dirname(os.path.abspath(__file__))

health_data_path = os.path.join(base_dir, "real_time_astronaut_health.csv")
stock_data_path = os.path.join(base_dir, "limited_stock_astronaut_food_calories.csv")
cabin_data_path = os.path.join(base_dir, "cabin_environment_data.csv")
cabin_data = pd.read_csv(cabin_data_path)
data = pd.read_csv(health_data_path)
stock_data = pd.read_csv(stock_data_path)

stock_data.rename(columns={
    "Food Item": "Item",
    "Calories per 100g": "Calories",
    "Quantity Available (g)": "Quantity"
}, inplace=True)

THRESHOLDS = {
    "Heart Rate (BPM)": (50, 100),  
    "Blood Pressure (Systolic) (mmHg)": (90, 130),  
    "Blood Pressure (Diastolic) (mmHg)": (60, 85),
    "Oxygen Saturation (SpO₂, %)": (94, 100),  
    "Core Body Temperature (°C)": (36.1, 37.8),  
    "Respiration Rate (BPM)": (10, 20)  
}

CABIN_THRESHOLDS = {
    "Cabin Pressure (kPa)": (70, 101),  
    "Oxygen Levels (%)": (19.5, 23.5),  
    "CO₂ Levels (mmHg)": (0.3, 5.3),  
    "Temperature (°C)": (18, 27),  
    "Humidity Levels (%)": (30, 60), 
    "Radiation Levels (μSv/hour)": (0, 0.5),  
    "Toxic Gas Levels (ppm)": (0, 0.1)  
}

def speak_alerts(alerts):
    alert_message = " ".join(alerts)
    print(alert_message)
    engine.say(alert_message)
    engine.runAndWait()

def generate_diet_plan(stock_data, num_days, num_people):
    daily_calorie_need_per_person = 2500
    total_daily_need = daily_calorie_need_per_person * num_people
    total_required_calories = total_daily_need * num_days

    available_total_calories = sum((row['Calories'] / 100) * row['Quantity'] for index, row in stock_data.iterrows())
    max_possible_days = int(available_total_calories // total_daily_need)

    if max_possible_days < num_days:
        status = f"⚠️ Food stock is insufficient for {num_days} days. It will last only for {max_possible_days} days.\n\n"
    else:
        status = f"✅ Food stock is sufficient for {num_days} days.\n\n"

    prompt = f"""You are a space nutritionist. Create an optimized diet plan for {num_people} astronauts to efficiently manage available stock for {num_days} days.

Each astronaut requires approximately {daily_calorie_need_per_person} kcal per day, meaning the crew needs {total_daily_need} kcal per day, totaling {total_required_calories} kcal over {num_days} days.

The following food items with calorie values per 100g and total available quantity are:

"""

    for index, row in stock_data.iterrows():
        available_calories = (row['Calories'] / 100) * row['Quantity']
        prompt += f"- {row['Item']}: {row['Calories']} kcal per 100g, Quantity Available: {row['Quantity']}g (Total Calories: {available_calories:.2f})\n"

    prompt += f"""

Objective:
Prepare a clear, simple daily diet plan per astronaut. For each food item, mention:
- Exact quantity (in grams) per astronaut per day
- Approximate calories it will provide
Make sure the total calories for each astronaut is approximately {daily_calorie_need_per_person} kcal/day.
Ensure nutritional balance (carbs, proteins, fats, vitamins).
Do not show tables. Write in clear sentences like:
"Each astronaut should eat 200g of freeze-dried bananas, 100g of spinach, 150g of scrambled eggs..."
"""

    api_url = "https://llama-33-70b-instruct-turbo.taitanapi.workers.dev/?prompt=" + requests.utils.quote(prompt)

    try:
        response = requests.get(api_url, timeout=40)
        if response.status_code == 200:
            suggestions = response.json().get("response", "No suggestions available.")
        else:
            suggestions = "Error: Unable to fetch diet plan from API."
    except requests.RequestException as e:
        suggestions = f"Error: API request failed. {e}"

    return status + suggestions

def generate_health_fix(row):
    prompt = f"""
You are a medical AI monitoring astronauts' health vitals in space. Analyze the following health data recorded at {row['Timestamp']}:

- Heart Rate: {row['Heart Rate (BPM)']} BPM
- Blood Pressure: {row['Blood Pressure (Systolic) (mmHg)']}/{row['Blood Pressure (Diastolic) (mmHg)']} mmHg
- Oxygen Saturation (SpO₂): {row['Oxygen Saturation (SpO₂, %)']}%
- Core Body Temperature: {row['Core Body Temperature (°C)']}°C
- Respiration Rate: {row['Respiration Rate (BPM)']} BPM

Rules:
1. If all parameters are normal, reply: "All vitals normal. No action needed."
2. If any parameter is abnormal, provide a short, clear one-liner fix.
3. If multiple parameters are abnormal, combine advice into one short, actionable alert.
4. If critical, mention "Critical: Immediate attention needed."
5. Keep it concise (max 2 lines).

Reply with only the final fix, nothing else.
"""

    api_url = "https://llama-33-70b-instruct-turbo.taitanapi.workers.dev/?prompt=" + requests.utils.quote(prompt)

    try:
        response = requests.get(api_url, timeout=40)
        if response.status_code == 200:
            return response.json().get("response", "No health fix available.")
        else:
            return "Error: Unable to fetch health fix."
    except requests.RequestException as e:
        return f"Error: API request failed. {e}"

def analyze_data(row):
    alerts = []
    oxygen_issue = False

    for col, (low, high) in THRESHOLDS.items():
        value = row[col]
        if value < low:
            alerts.append(f"{col} is too low: {value}")
            if col == "Oxygen Saturation (SpO₂, %)":
                oxygen_issue = True
        elif value > high:
            alerts.append(f"{col} is too high: {value}")

    if alerts:
        print(f"\n🚨 ALERT: {row['Timestamp']} - Critical health parameters detected:")
        for alert in alerts:
            print(f" - {alert}")

        if oxygen_issue:
            print(f"\n⚠️ Warning: Oxygen saturation is low for astronaut at {row['Timestamp']}. Ensure collective oxygen supply and ventilation is adequate.")

    health_fix = generate_health_fix(row)
    
    # Speak all alerts together
    alerts.append(f"🩺 Quick Health Advice: {health_fix}")
    speak_alerts(alerts)

def generate_cabin_fix(row):
    prompt = f"""
You are a space station environment AI. Analyze the following cabin environment data recorded at {row['Timestamp']}:

- Cabin Pressure: {row['Cabin Pressure (kPa)']} kPa
- Oxygen Levels: {row['Oxygen Levels (%)']} %
- CO₂ Levels: {row['CO₂ Levels (mmHg)']} mmHg
- Temperature: {row['Temperature (°C)']} °C
- Humidity Levels: {row['Humidity Levels (%)']} %
- Radiation Levels: {row['Radiation Levels (μSv/hour)']} μSv/hour
- Toxic Gas Levels: {row['Toxic Gas Levels (ppm)']} ppm

Rules:
1. If all parameters are normal, reply: "Cabin environment stable. No action needed."
2. If any parameter is abnormal, provide a short, clear one-liner fix.
3. If multiple parameters are abnormal, combine advice into one short, actionable alert.
4. If critical, mention "Critical: Immediate action required."
5. Keep it concise (max 2 lines).

Reply with only the final fix, nothing else.
"""

    api_url = "https://llama-33-70b-instruct-turbo.taitanapi.workers.dev/?prompt=" + requests.utils.quote(prompt)

    try:
        response = requests.get(api_url, timeout=40)
        if response.status_code == 200:
            return response.json().get("response", "No cabin environment fix available.")
        else:
            return "Error: Unable to fetch cabin environment fix."
    except requests.RequestException as e:
        return f"Error: API request failed. {e}"

def analyze_cabin_data(row):
    alerts = []
    critical_issue = False

    for col, (low, high) in CABIN_THRESHOLDS.items():
        value = row[col]
        if value < low:
            alerts.append(f"{col} too low: {value}")
            if col in ["Cabin Pressure (kPa)", "Oxygen Levels (%)", "Toxic Gas Levels (ppm)"]:
                critical_issue = True
        elif value > high:
            alerts.append(f"{col} too high: {value}")
            if col in ["Cabin Pressure (kPa)", "Radiation Levels (μSv/hour)", "Toxic Gas Levels (ppm)"]:
                critical_issue = True

    if alerts:
        print(f"\n🌐 ALERT: {row['Timestamp']} - Cabin environment issue detected:")
        for alert in alerts:
            print(f" - {alert}")
        if critical_issue:
            print(f"\n⚠️ Critical: Immediate attention required for cabin environment at {row['Timestamp']}.")

        speak_alerts(alerts)

    cabin_fix = generate_cabin_fix(row)
    print(f"\n🛠️ Cabin Quick Advice: {cabin_fix}")
    print("=" * 80)

# --- Simultaneous Monitoring Function ---
def simultaneous_monitoring(interval=10):
    max_rows = max(len(data), len(cabin_data))
    
    for i in range(max_rows):
        if i < len(data):
            print(f"\nProcessing health data at {data.loc[i, 'Timestamp']}...")
            analyze_data(data.loc[i])
        if i < len(cabin_data):
            print(f"\nProcessing cabin data at {cabin_data.loc[i, 'Timestamp']}...")
            analyze_cabin_data(cabin_data.loc[i])
        
        time.sleep(interval)

def analyze_diet_plan():
    diet_plan = generate_diet_plan(stock_data, num_days, num_people)
    print(f"\n📋 Optimized Diet Plan for {num_days} Days & {num_people} Astronauts:\n{diet_plan}")

if __name__ == "__main__":

    while True:
        try:
            num_days = int(input("Enter the number of days to manage with the current stock: "))
            if num_days <= 0:
                print("Please enter a valid positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    while True:
        try:
            num_people = int(input("Enter the number of astronauts: "))
            if num_people <= 0:
                print("Please enter a valid positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    print("\n🍽️ Generating Diet Plan Analysis...\n")
    analyze_diet_plan()

    print("\n🔍 Starting Real-time Health & Cabin Monitoring...\n")
    simultaneous_monitoring(interval=10)


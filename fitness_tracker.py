import streamlit as st
from datetime import datetime
import pytz
import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Hide warnings
warnings.filterwarnings('ignore')

 # 🌟 UI Styling - Improved background and font colors
st.markdown(
    """
    <style>
    /* Main app background (right side) */
    .stApp {
        background-color: #F5F7FA;  /* Light grey/blue background */
        color: #000000;  /* Black text */
    }

    /* Sidebar background (left side) */
    section[data-testid="stSidebar"] {
        background-color: #2E3B55;  /* Navy blue sidebar */
        color: white;
    }

    /* All text inside sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Custom styling for big-font (prediction result) */
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #FF4B4B;
    }

    /* Improve overall font visibility */
    h1, h2, h3, h4, h5, h6, p {
        color: #000000;
    }

    /* Override widget label color in main area */
    .css-10trblm, .css-1d391kg {
        color: #000000 !important;
    }

    /* Optional: Make sidebar header bold */
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2 {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 📚 File Paths
calories_file = r"calories.csv"
exercise_file = r"exercise.csv"

# Function to get current IST time
def get_current_time():
    india_tz = pytz.timezone('Asia/Kolkata')
    return datetime.now(india_tz).strftime("%I:%M:%S %p")

# Sidebar live clock
st.sidebar.markdown("## 🕒 Live Clock")
clock_placeholder = st.sidebar.empty()

# Update the live clock
current_time = get_current_time()
clock_placeholder.markdown(
    f"<h2 style='text-align: center;'>{current_time}</h2>", unsafe_allow_html=True
)

# --- Main App Logic Starts Here ---
# ❗ Check if files exist
if not os.path.exists(calories_file) or not os.path.exists(exercise_file):
    st.error("Error: Required CSV files (`calories.csv`, `exercise.csv`) are missing!")
    st.stop()

# 🔄 Load Data
@st.cache_data
def load_data():
    calories = pd.read_csv(calories_file)
    exercise = pd.read_csv(exercise_file)
    data = exercise.merge(calories, on="User_ID")  # Ensure 'User_ID' exists in both files
    return data

data = load_data()

# 🏗 Convert Categorical Data
data = pd.get_dummies(data, drop_first=False)

# 🏷 Detect Gender Column Dynamically
gender_cols = [col for col in data.columns if 'Gender' in col]
if not gender_cols:
    st.error("No gender-related column found after encoding!")
    st.stop()

gender_column = gender_cols[0]  # Use the first detected gender column

# 🔗 SQLite Database
conn = sqlite3.connect("fitness_tracker.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, 
        age INTEGER, 
        gender TEXT, 
        height REAL, 
        weight REAL, 
        bmi REAL, 
        calories REAL)
''')
conn.commit()

# 🎨 Streamlit UI
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.write(" In This WebApp you will able to observe your predicted colaries burned in your body.Pass your parameter like 'AGE', 'GENDER', 'BMI' and etc ,into this web app and then you will see the predicted value of kilocalories burned in your body.")
st.write("Track your calories burned based on your health data.")

st.sidebar.header("📊 User Input Parameters")

# 📥 User Input Function
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    height = st.sidebar.slider("Height (cm)", 140, 200, 170)
    weight = st.sidebar.slider("Weight (kg)", 40, 120, 70)
    duration = st.sidebar.slider("Workout Duration (min)", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    
    gender = 1 if gender_button == "Male" else 0  # Convert to binary
    bmi = round(weight / ((height / 100) ** 2), 2)

    return pd.DataFrame({
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender": [gender_button]  # Store as text in DB
    })

df = user_input_features()

# 🔎 Display User Input
st.write("### 📝 Your Input Data")
st.write(df)

# 📤 Save User Data to Database
cursor.execute("INSERT INTO users (age, gender, height, weight, bmi, calories) VALUES (?, ?, ?, ?, ?, ?)", 
               (df["Age"].values[0], df["Gender"].values[0], df["Height"].values[0], df["Weight"].values[0], df["BMI"].values[0], 0))
conn.commit()

# 📊 Machine Learning Model Training
data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
feature_columns = gender_cols + ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
data = data[feature_columns]

X = data.drop("Calories", axis=1)
y = data["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 📈 Improve Model with Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔥 Gradient Boosting Model for Better Accuracy
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train_scaled, y_train)

# 🎯 Align Input Data for Prediction
df_ml = df.copy()
df_ml["Gender"] = 1 if df_ml["Gender"].values[0] == "Male" else 0
df_ml = df_ml.reindex(columns=X_train.columns, fill_value=0)

df_ml_scaled = scaler.transform(df_ml)
prediction = model.predict(df_ml_scaled)

# 🔥 Display Prediction
st.markdown(f"<p class='big-font'>🔥 Predicted Calories Burned: {round(prediction[0], 2)} kcal</p>", unsafe_allow_html=True)

# 🔄 Update Database
cursor.execute("UPDATE users SET calories = ? WHERE id = (SELECT MAX(id) FROM users)", (round(prediction[0], 2),))
conn.commit()

# 📊 Show Similar Results
st.write("### 📍 Similar Results from Dataset")
similar_data = data[(data["Calories"] >= prediction[0] - 10) & (data["Calories"] <= prediction[0] + 10)]
st.write(similar_data.sample(min(5, len(similar_data))))  # Avoid error if fewer than 5 results exist

# 📈 Fitness Trends
st.write("### 📊 Fitness Trends Over Age")

fig, ax = plt.subplots()
sns.scatterplot(data=data, x="Age", y="Calories", hue=gender_column, ax=ax)
st.pyplot(fig)

# 📜 Personalized Health Insights
st.write("### 🩺 Health Advisory & Feedback")

# Evaluate BMI and provide impactful feedback
if df["BMI"].values[0] < 18.5:
    st.write("🚨 **Alert: Your BMI is critically low!** Being underweight can weaken your immune system, increase fatigue, and lead to deficiencies.")
    st.write("**🌟 Here's what you can do:**")
    st.write("- 🍔 Eat energy-dense meals like nuts, dried fruits, and avocados.")
    st.write("- 🥚 Incorporate proteins like eggs, fish, and dairy to help rebuild muscle.")
    st.write("- 🩺 Consult a doctor to check for underlying causes.")
elif 18.5 <= df["BMI"].values[0] <= 24.9:
    st.write("🎉 **Fantastic News! Your BMI is in the healthy range!** Did you know a healthy BMI reduces your risk of chronic diseases like diabetes and hypertension?")
    st.write("**✨ Keep it up with these habits:**")
    st.write("- 🥗 Maintain a balance of proteins, carbs, and fats in your diet.")
    st.write("- 🚴‍♀️ Keep up with regular exercise like 30 minutes of walking or cycling daily.")
    st.write("- 🍰 Celebrate with cheat meals occasionally—but don’t overdo it!")
elif df["BMI"].values[0] >= 25:
    st.write("⚠️ **Caution: Your BMI indicates you're overweight.** This increases the risk of heart disease, diabetes, and joint problems.")
    st.write("**🌟 Make these changes starting today:**")
    st.write("- 🥤 Swap sugary drinks for water or green tea.")
    st.write("- 🥬 Include more fiber-rich foods like vegetables, oats, and legumes.")
    st.write("- 🏃 Get at least 150 minutes of moderate physical activity weekly (e.g., brisk walking or swimming).")

# Check heart rate for actionable feedback
if df["Heart_Rate"].values[0] > 100:
    st.write("💓 **Warning: High Heart Rate Detected!** A consistently elevated heart rate can indicate stress, dehydration, or cardiovascular issues.")
    st.write("**🌟 Suggestions for a calmer heart:**")
    st.write("- 🧘 Take deep breaths for 5 minutes to activate relaxation.")
    st.write("- 💧 Stay hydrated and avoid caffeine for the day.")
    st.write("- 🩺 If your heart rate remains high, seek medical attention.")
elif df["Heart_Rate"].values[0] < 60:
    st.write("💓 **Low Heart Rate Alert:** Your heart rate is lower than normal, which could lead to dizziness or fatigue.")
    st.write("**🌟 Suggestions to energize:**")
    st.write("- 🏃‍♀️ Get up and stretch to boost circulation.")
    st.write("- 🍎 Eat a light snack to maintain energy levels.")
    st.write("- 🩺 Monitor for prolonged low heart rate and consult a physician if needed.")

# Evaluate body temperature for health recommendations
if df["Body_Temp"].values[0] > 37.5:
    st.write("🌡️ **Alert: Elevated Body Temperature Detected!** This could signal fever, dehydration, or an infection.")
    st.write("**🌟 Immediate actions to take:**")
    st.write("- 💧 Drink plenty of fluids to reduce dehydration.")
    st.write("- 🛌 Rest in a cool, well-ventilated environment.")
    st.write("- 🩺 If your fever persists, visit a doctor promptly.")
elif df["Body_Temp"].values[0] < 35:
    st.write("🌡️ **Caution: Low Body Temperature!** Prolonged exposure to cold or poor circulation might be causing this.")
    st.write("**🌟 Here's how you can warm up:**")
    st.write("- 🧣 Layer up with warm clothing or blankets.")
    st.write("- ☕ Drink warm liquids like herbal tea or soup.")
    st.write("- 🩺 Monitor closely and seek help if symptoms persist.")

# General recommendations for all users
st.write("💧 **Stay Hydrated:** Did you know dehydration can cause headaches, fatigue, and low focus? Make sure to drink at least 8 glasses of water daily!")
st.write("🍽️ **Healthy Eating Reminder:** Skipping meals can slow down your metabolism. Eat on time to fuel your body and mind effectively!")
st.write("🌙 **Sleep Matters:** Aim for 7-8 hours of quality sleep. Your body needs rest to repair itself and stay energized.")
st.write("🚶‍♂️ **Move It:** Sitting too long can harm your health. Take a 5-minute break every hour to stretch or walk!")

  
# 📂 Show Past Records
users_df = pd.read_sql_query("SELECT * FROM users", conn)
st.write("### 📜 Past User Records")
st.write(users_df)

conn.close()

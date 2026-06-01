import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("course_dataset.csv")

# Prepare data
X = df.drop("course", axis=1)
y = df["course"]

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------- UI ---------------- #

st.set_page_config(page_title="Career Counselling System", layout="centered")

st.title("🎓 Career Counselling System")

st.markdown("Fill in your details to get course recommendations")

# -------- Personal Details -------- #
st.header("👤 Personal Details")

name = st.text_input("Full Name")
age = st.number_input("Age", 15, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# -------- Academic Details -------- #
st.header("📊 Academic Performance")

math = st.number_input("Math Score", 0, 100)
physics = st.number_input("Physics Score", 0, 100)
chemistry = st.number_input("Chemistry Score", 0, 100)
biology = st.number_input("Biology Score", 0, 100)
english = st.number_input("English Score", 0, 100)

# -------- Interests -------- #
st.header("💡 Interests & Skills (Rate 1–10)")

programming = st.slider("Programming Interest", 1, 10)
analytical = st.slider("Analytical Skills", 1, 10)
creativity = st.slider("Creativity Level", 1, 10)
communication = st.slider("Communication Skills", 1, 10)
leadership = st.slider("Leadership Skills", 1, 10)

# -------- Achievements -------- #
st.header("🏆 Achievements")

sports = st.selectbox("Sports Participation", ["No", "Yes"])
olympiad = st.selectbox("Olympiad Participation", ["No", "Yes"])
projects = st.slider("Projects Done", 0, 5)

# Convert categorical to numeric
sports = 1 if sports == "Yes" else 0
olympiad = 1 if olympiad == "Yes" else 0

# -------- Prediction -------- #
if st.button("🔍 Get Recommendation"):

    input_data = [[
        math, physics, chemistry, biology, english,
        programming, analytical, creativity,
        communication, leadership,
        sports, olympiad, projects
    ]]

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    course = le.inverse_transform(prediction)

    st.success(f"Recommended Course: {course[0]}")

    st.info(f"Good luck, {name}! ")
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Dataset
# ---------------------------

data = {
    "Name": ["ajay", "raj", "sonu"],
    "age": [24, 35, 42],
    "fasting_blood_sugar": [96, 105, 127],
    "HbA1c": [4.8, 5.5, 6.8],
    "diabetic": ["normal", "pre-diabetic", "diabetic"]
}

df = pd.DataFrame(data)

# Label encode
le = LabelEncoder()
df["diabetic_label"] = le.fit_transform(df["diabetic"])

X = df[["age", "fasting_blood_sugar", "HbA1c"]]
y = df["diabetic_label"]

# Model
model = RandomForestClassifier()
model.fit(X, y)

# ---------------------------
# 2. Streamlit UI
# ---------------------------

st.title("🩺 Diabetes Prediction App")
st.write("Fill patient details to predict diabetes status.")

age = st.number_input("Age", min_value=1, max_value=120)
fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300)
hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0)

if st.button("Predict"):
    new_data = [[age, fbs, hba1c]]
    result = model.predict(new_data)[0]
    output = le.inverse_transform([result])[0]
    
    st.subheader(f"Patient Status: {output.upper()}")
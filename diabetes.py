import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. DATAFRAME
# -----------------------------
data = {
    "Name": ["ajay", "raj", "sonu"],
    "age": [24, 35, 42],
    "fasting_blood_sugar": [96, 105, 127],
    "HbA1c": [4.8, 5.5, 6.8],
    "diabetic": ["normal", "pre-diabetic", "diabetic"]
}

df = pd.DataFrame(data)
print(df)

# -----------------------------
# 2. Label Encoding
# -----------------------------
le = LabelEncoder()
df["diabetic_label"] = le.fit_transform(df["diabetic"])

# -----------------------------
# 3. Features and target
# -----------------------------
X = df[["age", "fasting_blood_sugar", "HbA1c"]]
y = df["diabetic_label"]

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# 5. Train Random Forest Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 6. Accuracy
# -----------------------------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# -----------------------------
# 7. Predict new patient
# -----------------------------
print("\n--- Predict Diabetes for New Patient ---")

age = int(input("Enter age: "))
fbs = int(input("Enter fasting blood sugar: "))
hba1c = float(input("Enter HbA1c: "))

new_data = [[age, fbs, hba1c]]

result = model.predict(new_data)[0]
output = le.inverse_transform([result])[0]

print("\nPatient Status:", output.upper())
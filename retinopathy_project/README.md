# 👁️ Diabetic Retinopathy Detection

This project uses a trained deep learning model to classify **Retinopathy severity** from retina fundus images.

## 📌 Project Overview
Diabetic Retinopathy (DR) is a diabetes-related eye condition that can lead to blindness.  
Using AI, this model predicts DR severity from fundus images with 5 classes:

- **0 – No DR**
- **1 – Mild**
- **2 – Moderate**
- **3 – Severe**
- **4 – Proliferative DR**

## 🧠 Model Used
- TensorFlow / Keras CNN model  
- Trained on retina fundus images  
- Preprocessed to 224×224 resolution  
- Output → 5-class classification

## 📷 App Details
The Streamlit app allows you to:

- Upload a retina fundus image  
- View the uploaded image  
- Get predicted DR severity instantly  

### 🔗 App File
`app_ratinopathy.py`

### 🔗 Model File
`dr_model.h5`

---

## 🚀 How to Run the Project

### 1️⃣ Install requirements

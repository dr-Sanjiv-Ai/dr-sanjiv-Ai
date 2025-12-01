import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("dr_model.h5")

st.title("👁 Diabetic Retinopathy Detection App")
st.write("Upload a retina image to predict DR severity (0 to 4).")

uploaded_file = st.file_uploader("Upload retina image", type=["png", "jpg", "jpeg"])

class_names = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative DR"
]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.subheader("🔍 Prediction:")
    st.success(class_names[pred_class])
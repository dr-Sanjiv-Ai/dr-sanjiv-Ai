import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("my_model.h5")

st.title("Pneumonia Detection From Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict if it is NORMAL or PNEUMONIA.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    if prob > 0.5:
        st.error(f"❌ RESULT: **PNEUMONIA**  (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✔️ RESULT: **NORMAL**  (Confidence: {(1-prob)*100:.2f}%)")
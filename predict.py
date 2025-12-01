import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("my_model.h5")

# Image path
img_path = "chest1.jpeg"  # EXACT SAME NAME

# Load image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("RESULT: PNEUMONIA")
else:
    print("RESULT: NORMAL")
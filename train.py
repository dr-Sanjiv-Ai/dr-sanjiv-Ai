import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# ---------------------------------------
# 1. PATHS (Make sure dataset folder is created by sort_image.py)
# ---------------------------------------
train_dir = "dataset/train"     # yaha 0,1,2,3,4 folders honge
val_dir   = "dataset/train"     # simple validation split yahi se ले लेगा

# ---------------------------------------
# 2. IMAGE GENERATOR
# ---------------------------------------
img_size = 224
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation"
)

# ---------------------------------------
# 3. MODEL (EfficientNetB0)
# ---------------------------------------
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False  # Freeze base model

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(5, activation='softmax')(x)  # 5 classes (0–4)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------
# 4. TRAIN MODEL
# ---------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5   # doctor sahab, aap 10–20 epochs bhi kar sakte ho
)

# ---------------------------------------
# 5. SAVE MODEL
# ---------------------------------------
model.save("dr_model.h5")

print("Training Complete! Model saved as dr_model.h5")
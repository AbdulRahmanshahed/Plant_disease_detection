# predict_image.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = tf.keras.models.load_model("plant_model.h5")

# Recreate the class labels from training directory
train_dir = "dataset/train"
class_labels = sorted(os.listdir(train_dir))  # all folder names = classes

print("🌿 PlantVillage Classifier Ready! Type 'exit' to quit.\n")

while True:
    img_path = input("Enter image path (from test folder): ")

    if img_path.lower() == "exit":
        print("👋 Exiting classifier.")
        break

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        print(f"🌱 Prediction: {predicted_class}\n")

    except Exception as e:
        print(f"⚠️ Error: {e}\n")

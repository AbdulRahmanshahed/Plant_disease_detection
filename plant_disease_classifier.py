# plant_classifier.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Dataset directories
train_dir =  r"C:\Users\DELL\OneDrive\Desktop\AI&Ml internship\Week 2 task\task 2"

# 2. Image settings
img_size = (128, 128)
batch_size = 32

# 3. Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20% for validation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Automatically detect number of classes
num_classes = train_generator.num_classes
print(f"✅ Number of classes detected: {num_classes}")

# 4. CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')   # dynamic classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
history = model.fit(train_generator, validation_data=val_generator, epochs=12)

# 6. Save model
model.save("plant_model.h5")
print("✅ Model saved as plant_model.h5")

# 7. Plot training curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.show()

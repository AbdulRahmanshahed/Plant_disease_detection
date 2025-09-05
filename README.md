# 🌿 PlantVillage Disease Classifier (CNN with TensorFlow)

This project is part of my AI/ML internship (Week 2). It builds a Convolutional Neural Network (CNN) to classify plant diseases from the **PlantVillage dataset**, which contains multiple crop species (Apple, Potato, Tomato, etc.) with healthy and diseased leaf images. The dataset is organized into two folders: `train/` (multiple subfolders, one per class) and `test/` (unlabeled images for prediction). The workflow: load images → resize (128x128) → normalize pixels → train CNN (Conv2D → ReLU → MaxPool → Dense → Softmax) → evaluate accuracy → save model (`plant_model.h5`) → predict test images. Tech stack: Python, TensorFlow/Keras, NumPy, Matplotlib. Files: `plant_classifier.py` (training script), `predict_image.py` (interactive prediction), `plant_model.h5` (trained model).  

### ⚡ How to Run
```bash
pip install tensorflow matplotlib scikit-learn numpy
python plant_classifier.py     # trains and saves model
python predict_image.py        # test with images from test/ folder
```
### 📊 Example Usage
Enter image path: dataset/test/img_1.jpg
🌱 Prediction: Tomato__Late_blight

Enter image path: dataset/test/img_2.jpg
🌱 Prediction: Apple__Black_rot

Enter image path: exit
👋 Exiting classifier.

###📈 Results

CNN trained for 12 epochs

Achieved high accuracy (~90%+ depending on dataset split)

Training & validation accuracy curves plotted during training

Works with any number of classes automatically (detected from train/ folders)

📜 Open-source for educational purposes.

---

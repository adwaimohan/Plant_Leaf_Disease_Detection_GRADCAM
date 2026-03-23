Here is the **clean, fully copyable README (no formatting issues):**

---

# 🌿 Crop Disease Detection using EfficientNet-B0 + Grad-CAM

A user-friendly web application that detects plant diseases from leaf images and visualizes model attention using Grad-CAM. Built with PyTorch and Streamlit.

---

## 📂 Project Structure

crop-disease-detection/
├── app.py                  # Streamlit web app for prediction + Grad-CAM
├── split.py                # Script to split dataset into train/val/test
├── requirements.txt        # Required dependencies
├── dataset/                # (Not included) PlantVillage dataset
├── disease_detection_model.pth  # (Not included) Trained model weights
└── README.md

---

## 📸 Screenshot
<img width="500" height="598" alt="image" src="https://github.com/user-attachments/assets/d4c7d358-10a7-4afd-a1f0-68ffdbe69ea1" />
<img width="470" height="383" alt="image" src="https://github.com/user-attachments/assets/993026c9-6aac-4469-9e9d-eaffb976c965" />



---

## ⚙️ Features

* Real-time crop disease classification
* Grad-CAM visualization for explainability
* Upload leaf images via web UI
* Confidence score for predictions
* Fast inference using EfficientNet-B0

---

## 🧰 Requirements

Make sure Python 3.8+ is installed. Then run:

pip install -r requirements.txt

Dependencies:

* torch
* torchvision
* streamlit
* numpy
* opencv-python
* Pillow

---

## 🚀 How to Use

1. Clone the repository

git clone [https://github.com/YOUR_USERNAME/crop-disease-detection.git](https://github.com/YOUR_USERNAME/crop-disease-detection.git)
cd crop-disease-detection

2. Download Dataset

Download the PlantVillage dataset and place it inside:

dataset/PlantVillage/

3. Add Trained Model

Place your trained model file:

disease_detection_model.pth

in the root directory

4. Run the App

streamlit run app.py

5. Use the Application

* Upload a leaf image (JPG/PNG)
* View predicted disease
* Check confidence score
* See Grad-CAM heatmap

---

## 📊 Dataset

* PlantVillage Dataset
* Contains labeled images of healthy and diseased plant leaves
* Organized into multiple disease classes

---

## 🧠 Model Details

* Architecture: EfficientNet-B0
* Framework: PyTorch
* Pretrained on ImageNet
* Fine-tuned for crop disease classification

---

## 🔍 How It Works

1. Image uploaded via Streamlit
2. Preprocessed (resize + normalization)
3. Passed through EfficientNet
4. Prediction using softmax
5. Grad-CAM highlights important regions

---


## 📖 Notes

* Dataset and model weights are not included due to size limits
* Runs on CPU by default (can modify for GPU)
* Best results with clear leaf images

---

## 🔮 Future Improvements

* Add more crop types
* Deploy online
* Add treatment suggestions
* Improve accuracy with better models

---

## ⭐ If you found this useful, give it a star!

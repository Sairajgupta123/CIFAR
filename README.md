# 🧠 CIFAR-10 Image Classification using Convolutional Neural Network (CNN)

A Deep Learning project that classifies images into one of the **10 CIFAR-10 object categories** using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**. The project includes a **Streamlit web application** that allows users to upload images and receive real-time predictions with confidence scores.

---

## 🚀 Features

- 🧠 CNN model trained on the CIFAR-10 dataset
- 📂 Upload an image for prediction
- 🎯 Predicts one of the 10 CIFAR-10 classes
- 📊 Displays prediction confidence
- 📈 Shows probability for all classes
- ⚡ Interactive Streamlit interface
- 💻 Easy to run locally

---

## 📂 CIFAR-10 Classes

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow

---

## 🧠 CNN Architecture

```
Input Image (32×32×3)
        │
        ▼
Conv2D (32 Filters)
        │
MaxPooling2D
        │
Conv2D (64 Filters)
        │
MaxPooling2D
        │
Conv2D (64 Filters)
        │
Flatten
        │
Dense (64)
        │
Dropout (0.5)
        │
Dense (10 - Softmax)
```

---

## 📈 Model Performance

| Metric | Result |
|---------|--------|
| Training Accuracy | ~78% |
| Validation Accuracy | ~73% |
| Test Accuracy | **71.82%** |

---

## 📁 Project Structure

```
CIFAR10-Image-Classifier/
│── app.py
│── cifar10_cnn.keras
│── CNN_CIFAR.ipynb
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/CIFAR10-Image-Classifier.git
```

### Move into the Project Directory

```bash
cd CIFAR10-Image-Classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📷 Application Preview

Add screenshots here:

- 🏠 Home Page
- 📂 Image Upload
- 🎯 Prediction Result
- 📊 Confidence Score
- 📈 Probability Graph

---

## 🎯 Future Improvements

- Improve model accuracy
- Data Augmentation
- Batch Normalization
- Transfer Learning (ResNet50 / MobileNetV2)
- Top-3 Predictions
- Deploy on Streamlit Cloud

---

## 👨‍💻 Author

**Sairaj Dilip Gupta**

- 🎓 B.Tech Information Technology
- 🤖 Deep Learning & Computer Vision Enthusiast

---

## ⭐ Support

If you like this project, please ⭐ the repository and share it with others.

---

## 📄 License

This project is developed for educational and learning purposes.

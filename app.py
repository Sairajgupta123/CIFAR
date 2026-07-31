import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_cnn.keras")

model = load_model()

# -----------------------------
# CIFAR-10 Classes
# -----------------------------
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# -----------------------------
# Title
# -----------------------------
st.title("🧠 CIFAR-10 Image Classification")
st.write("Upload an image and let the CNN model predict its class.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize to CIFAR size
    img = image.resize((32, 32))

    img_array = np.array(img).astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Prediction : **{class_names[predicted_class]}**")

    st.info(f"Confidence : **{confidence:.2f}%**")

    st.subheader("Prediction Probabilities")

    probs = prediction[0]

    for i, cls in enumerate(class_names):
        st.progress(float(probs[i]))
        st.write(f"{cls} : {probs[i]*100:.2f}%")
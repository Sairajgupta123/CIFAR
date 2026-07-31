import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_cnn.keras")

model = load_model()

classes = [
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

st.title("🧠 CIFAR-10 Image Classification")
st.write("Upload an image or capture one using your webcam.")

option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Capture Photo"]
)

image = None

# Upload
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# Camera
else:
    camera_photo = st.camera_input("Take a Photo")

    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")

# Prediction
if image is not None:

    st.image(image, caption="Selected Image", use_container_width=True)

    img = image.resize((32, 32))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    pred = np.argmax(prediction)
    confidence = prediction[pred] * 100

    st.success(f"Prediction: **{classes[pred]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Class Probabilities")

    for i in range(len(classes)):
        st.write(f"**{classes[i]}**")
        st.progress(float(prediction[i]))
        st.write(f"{prediction[i]*100:.2f}%")
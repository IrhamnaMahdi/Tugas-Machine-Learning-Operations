import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bird Species Classifier",
    page_icon="🦜",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bird_species_model_final.keras")

@st.cache_data
def load_labels():
    with open("class_indices.json") as f:
        labels = json.load(f)
    return {v: k.replace('_', ' ').title() for k, v in labels.items()}

model = load_model()
class_names = load_labels()

IMG_SIZE = 224

st.markdown("""
<h1 style='text-align: center;'>🦜 Bird Species Classifier</h1>
<p style='text-align: center; color: gray;'>Deep Learning · MobileNetV2 · TensorFlow</p>
<hr>
""", unsafe_allow_html=True)

st.subheader("📤 Upload Bird Image")
uploaded_file = st.file_uploader(
    "Upload an image (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing image..."):
        preds = model.predict(img_array)[0]

    top_idx = np.argmax(preds)
    confidence = preds[top_idx] * 100

    st.markdown(f"""
    <h2 style='text-align: center;'>✅ Prediction Result</h2>
    <h3 style='text-align: center; color: green;'>
        {class_names[top_idx]}
    </h3>
    <p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>
    """, unsafe_allow_html=True)

    st.subheader("📊 Prediction Confidence")

    sorted_idx = np.argsort(preds)[::-1]
    labels = [class_names[i] for i in sorted_idx]
    values = preds[sorted_idx] * 100

    fig, ax = plt.subplots()
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence (%)")
    st.pyplot(fig)

st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    **Bird Species Classifier**

    - CNN with **MobileNetV2**
    - Transfer Learning (ImageNet)
    - Trained on custom Kaggle dataset

    Classes:
    - Amazon Green Parrot
    - Gray Parrot
    - Macaw
    - White Parrot
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed with TensorFlow & Streamlit")
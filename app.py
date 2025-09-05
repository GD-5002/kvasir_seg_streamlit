import streamlit as st
from PIL import Image
import os
import tempfile
import requests
import tensorflow as tf
import numpy as np

# ----------------------
# STREAMLIT CONFIG
# ----------------------
st.set_page_config(page_title="Kvasir-SEG Polyp Detection", layout="centered")
st.title("Kvasir-SEG Polyp Detection (Keras)")

# ----------------------
# MODEL SETUP
# ----------------------
MODEL_PATH = "best_model.h5"
FILE_ID = "1azV2zxhTPzSSx13BK9nVO_x9aatTldzs"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Download model if not present or corrupted
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
    st.info("Downloading Keras model (~42 MB) from Google Drive...")
    try:
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

# Load model (cached for repeated use)
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ----------------------
# IMAGE UPLOAD
# ----------------------
uploaded_file = st.file_uploader("Upload an endoscopy image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        temp_image_path = tmp_file.name
        image.save(temp_image_path)

    # Preprocess image
    input_size = (256, 256)  # Adjust according to model input
    img_resized = image.resize(input_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------
    # PREDICTION
    # ----------------------
    with st.spinner("Detecting polyps..."):
        pred_mask = model.predict(img_array)[0]

        if pred_mask.shape[-1] == 1:
            pred_mask = np.squeeze(pred_mask, axis=-1)

        mask_img = (pred_mask * 255).astype(np.uint8)
        mask_img_pil = Image.fromarray(mask_img)
        mask_img_pil = mask_img_pil.resize(image.size)

    # ----------------------
    # DISPLAY RESULTS
    # ----------------------
    st.subheader("Predicted Mask")
    st.image(mask_img_pil, use_column_width=True)

    overlay = image.copy()
    overlay_array = np.array(overlay)
    overlay_array[pred_mask > 0.5] = [255, 0, 0]  # Highlight polyps in red
    overlay_img = Image.fromarray(overlay_array)

    st.subheader("Overlay on Original Image")
    st.image(overlay_img, use_column_width=True)

    os.remove(temp_image_path)

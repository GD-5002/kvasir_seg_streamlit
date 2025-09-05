import streamlit as st
from PIL import Image
import os
import tempfile
from ultralytics import YOLO
import gdown

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Kvasir-SEG Polyp Detection", layout="centered")
st.title("Kvasir-SEG Polyp Detection (YOLOv11)")

# Path to model
MODEL_PATH = "best_model.h5"

# Optional: Download from Google Drive if model not present
if not os.path.exists(MODEL_PATH):
    MODEL_DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with your file ID
    st.info("Downloading YOLOv11 model...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# Load YOLO model
@st.cache_resource(show_spinner=True)
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# ----------------------
# UPLOAD IMAGE
# ----------------------
uploaded_file = st.file_uploader("Upload an endoscopy image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        temp_image_path = tmp_file.name
        image.save(temp_image_path)

    # ----------------------
    # PREDICTION
    # ----------------------
    with st.spinner("Detecting polyps..."):
        results = model(temp_image_path)

    # Display results
    result_img = results[0].plot()
    st.image(result_img, caption="Detected Polyps", use_column_width=True)

    # Show detection details
    st.subheader("Detection Details")
    if results[0].boxes.data.shape[0] == 0:
        st.write("No polyps detected.")
    else:
        for det in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, cls = det
            st.write(f"Class: {int(cls)}, Confidence: {score:.2f}, BBox: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")

    # Clean up temp file
    os.remove(temp_image_path)

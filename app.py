import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="YOLOv11 Polyp Detection", layout="centered")
st.title("YOLOv11 Polyp Detection (Multiple Images)")

# ----------------------
# Load YOLOv11 model
# ----------------------
@st.cache_resource
def load_model():
    return YOLO("best_saved.pt")  # path to your YOLOv11 .pt model

model = load_model()

# ----------------------
# Image uploader
# ----------------------
uploaded_files = st.file_uploader(
    "Upload endoscopy images", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            temp_path = tmp_file.name
            image.save(temp_path)

        # ----------------------
        # Run YOLOv11 prediction
        # ----------------------
        results = model(temp_path, verbose=False)

        # Display YOLO predicted image
        if results[0].masks is not None or len(results[0].boxes) > 0:
            pred_img_array = results[0].plot()  # RGB numpy array
            pred_img = Image.fromarray(pred_img_array)
            st.image(pred_img, caption="YOLO Prediction", use_column_width=True)
        else:
            st.info("No polyps detected.")

        # Display detected polyps with confidence
        if len(results[0].boxes) > 0:
            st.subheader("Detected Polyps:")
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                st.write(f"- Class: {model.names[int(cls)]}, Confidence: {conf:.2f}")

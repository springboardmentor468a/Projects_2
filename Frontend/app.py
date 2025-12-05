import streamlit as st
from PIL import Image
import numpy as np
import torch
import io
import os
import base64

from model import load_model, preprocess, get_segmented_output

st.set_page_config(page_title="VisionExtract AI", layout="wide")

# -------------------------
# Helper: convert PIL image to base64 for HTML
# -------------------------
def img_to_bytes(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_segmentation_model():
    return load_model("final_segmentation_model.pth")

model = load_segmentation_model()

# -------------------------
# Custom CSS for colorful style
# -------------------------
st.markdown("""
<style>
/* Page background gradient */
body, .main {
    background: linear-gradient(135deg, #e0f7fa, #ffe0b2, #f8bbd0);
}

/* Image cards */
.card {
    background: linear-gradient(135deg, #ffffff, #f1f8e9);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

/* Sub-headings */
h4 {
    color: #333333;
}

/* Download button */
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #81C784);
    color: #ffffff;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #81C784, #4CAF50);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Gradient Header (Green)
# -------------------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #4CAF50, #81C784);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
">
    <h1 style="color: white; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); font-family: 'Segoe UI', sans-serif;">
        VisionExtract – Image Segmentation
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

IMG_SIZE = 320

# -------------------------
# Sample Section
# -------------------------
sample_path = "sample.jpg"

if os.path.exists(sample_path):
    sample = Image.open(sample_path).convert("RGB")
    _, sample_output = get_segmented_output(model, sample)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div class='card'>
            <h4>Sample Original</h4>
            <img src='data:image/png;base64,{img_to_bytes(sample)}' width='{IMG_SIZE}'>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <h4>Sample Output</h4>
            <img src='data:image/png;base64,{img_to_bytes(sample_output)}' width='{IMG_SIZE}'>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("sample.jpg not found in folder.")

st.markdown("---")

# -------------------------
# User Upload Section
# -------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    _, output_img = get_segmented_output(model, image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div class='card'>
            <h4>Your Original</h4>
            <img src='data:image/png;base64,{img_to_bytes(image)}' width='{IMG_SIZE}'>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <h4>Segmentation Output</h4>
            <img src='data:image/png;base64,{img_to_bytes(output_img)}' width='{IMG_SIZE}'>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # Centered Download Button
    # -------------------------
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.download_button(
            label="Download Output",
            data=buf.getvalue(),
            file_name="segmented_output.png",
            mime="image/png"
        )

else:
    st.info("Upload an image to get results")
import io
from pathlib import Path
from PIL import Image
import streamlit as st
import torch

from backend.model_utils import load_model, segment_object_only

# ----------------- Helpers -----------------
def resize_image(img: Image.Image, max_width=350):
    w, h = img.size
    if w <= max_width:
        return img
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)

# ----------------- Streamlit Setup -----------------
st.set_page_config(page_title="Pixiel Mask", layout="wide")

# ----------------- CSS Styling -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* Soft baby pink background */
[data-testid="stAppViewContainer"] { 
    background: linear-gradient(135deg, #ffd6eb, #ffe9f7, #ffffff);
}

/* Title */
.title { 
    font-size: 46px; 
    font-weight: 800; 
    text-align: center;
    background: linear-gradient(90deg, #ff70c6, #ff94d6, #ff7ab8);
    -webkit-background-clip: text; 
    color: transparent; 
    margin-bottom: -5px;
}

/* Subtitle */
.subtitle { 
    text-align: center; 
    font-size: 20px; 
    color: #8e4b7f; 
    margin-bottom: 40px; 
}

/* Cloud-style card */
.card { 
    background: rgba(255, 255, 255, 0.85); 
    padding: 28px; 
    border-radius: 25px; 
    backdrop-filter: blur(12px);
    box-shadow: 0px 6px 25px rgba(255, 150, 200, 0.25); 
    margin-bottom: 35px; 
    border-left: 6px solid #ff9cd4;
}

/* Section header */
.section-header { 
    font-size: 28px; 
    font-weight: 700; 
    background: linear-gradient(90deg, #b84a9a, #ff4fb3);
    -webkit-background-clip: text; 
    color: transparent; 
}

/* Feature boxes */
.feature-box { 
    background: #ffe2f1; 
    padding: 20px; 
    border-radius: 18px; 
    text-align: center; 
    box-shadow: 0 4px 18px rgba(255, 140, 180, 0.25); 
    transition: 0.25s; 
}
.feature-box:hover { 
    transform: translateY(-4px); 
    background: #ffd0eb; 
}
.feature-title { 
    font-size: 20px; 
    font-weight: 700; 
    margin-top: 10px; 
}

/* VIOLET BUTTONS */
.stButton > button { 
    background: linear-gradient(90deg,#7b2fff,#b983ff); 
    color: white; 
    font-size: 18px; 
    padding: 10px 25px; 
    border-radius: 14px; 
    border: none; 
    transition: 0.2s; 
}
.stButton > button:hover { 
    transform: scale(1.05); 
}

/* Violet download button */
.stDownloadButton > button { 
    background: linear-gradient(90deg,#6a0dad,#a56eff); 
    color: white; 
    font-size: 18px; 
    padding: 10px 25px; 
    border-radius: 14px; 
    border: none; 
}
.stDownloadButton > button:hover { 
    transform: scale(1.05); 
}

/* Image styling */
img { 
    border-radius: 14px; 
    box-shadow: 0 4px 16px rgba(255, 150, 200, 0.35); 
}
</style>
""", unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown("<div class='title'> Pixiel Mask</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Make your photos dreamy & beautiful with soft masking magic ✨</div>", unsafe_allow_html=True)

# ----------------- Features -----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🌸 Sweet Features</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""<div class='feature-box'><div style='font-size:45px;'>✨</div>
                   <div class='feature-title'>Soft Cutout</div>
                   Clean and smooth edges for perfect results.</div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class='feature-box'><div style='font-size:45px;'>💗</div>
                   <div class='feature-title'>Aesthetic Edit</div>
                   Gentle pastel tones for dreamy looks.</div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class='feature-box'><div style='font-size:45px;'>⚡</div>
                   <div class='feature-title'>Fast Processing</div>
                   Lightning-fast object masking.</div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Try Me Section -----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📸 Try Here </div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload your beautiful picture 💕", type=["png","jpg","jpeg","webp"])
bg_color = st.color_picker("Choose a soft background color 🌈", "#ffcce8")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(resize_image(img), caption="Your Photo 🌟", width=350)

    if st.button("Process Image 💜"):
        with st.spinner("Creating your aesthetic edit… ✨"):
            bg = tuple(int(bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            result = segment_object_only(img, model, device, bg)

        st.image(resize_image(result), caption="Aesthetic Output 💖", width=350)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Your Edit 🎀", buf, "pink_mask_output.png", "image/png")

st.markdown("</div>", unsafe_allow_html=True)

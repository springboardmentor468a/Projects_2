import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import base64

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Settings", layout="wide")

# -----------------------------------------------------
# THEME COLORS
# -----------------------------------------------------
STEEL = "#9CC3E5"
DARK = "#0B1220"

theme = st.session_state.get("theme", "light")
BG = DARK if theme == "dark" else STEEL
TEXT = "#FFFFFF" if theme == "dark" else "#000000"

# -----------------------------------------------------
# STYLE + NAVBAR
# -----------------------------------------------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {BG};
    color: {TEXT};
}}

.topmenu {{
  display:flex;
  justify-content:center;
  gap:25px;
  background: rgba(255,255,255,0.18);
  padding:14px;
  border-radius:12px;
  margin-bottom: 25px;
  backdrop-filter: blur(8px);
}}

.topmenu a {{
  font-size:18px;
  font-weight:700;
  color:{TEXT};
  text-decoration:none !important;
  padding:8px 16px;
  border-radius:8px;
  transition:0.25s;
}}

.topmenu a:hover {{
  background: rgba(255,255,255,0.28);
}}

.card {{
    background: rgba(255,255,255,0.22);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
}}

.action-btn {{
    background-color: #3498DB;
    color: white !important;
    padding: 10px 26px;
    border-radius: 30px;
    font-size:16px;
    font-weight:700;
    text-decoration:none !important;
    border:none !important;
    outline:none !important;
}}

.action-btn:hover {{
    background-color:#1F78C8;
}}

.btn-row {{
    display:flex;
    justify-content:center;
    gap:20px;
    margin-top: 18px;
}}
</style>

<div class="topmenu">
    <a href="/" target="_self">App</a>
    <a href="/about" target="_self">About</a>
    <a href="/history" target="_self">History</a>
    <a href="/settings" target="_self">Settings</a>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.markdown(f"<h1 style='text-align:center; color:{TEXT};'>Image Adjustment Settings</h1>", unsafe_allow_html=True)
st.write(" ")

# -----------------------------------------------------
# LOAD LATEST IMAGE
# -----------------------------------------------------
LATEST = "history/latest.png"

if not os.path.exists(LATEST):
    st.warning("No latest image found. Please extract an image from the App page first.")
    st.stop()

img = Image.open(LATEST).convert("RGB")

# -----------------------------------------------------
# LAYOUT
# -----------------------------------------------------
col_left, col_right = st.columns([1.4, 1])

# ===================== LEFT CONTROLS =====================
with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Adjustments Panel")

    brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
    contrast   = st.slider("Contrast", 0.0, 2.0, 1.0)
    sharpness  = st.slider("Sharpness", 0.0, 3.0, 1.0)

    blur_amt = st.slider("Blur Amount", 0, 30, 0)
    smooth_edges = st.slider("Smooth Edges", 0, 20, 0)
    opacity = st.slider("Background Opacity", 0.0, 1.0, 1.0)

    st.markdown("---")

    st.subheader("Crop & Rotate")
    crop_left = st.slider("Crop Left (%)", 0, 45, 0)
    crop_top = st.slider("Crop Top (%)", 0, 45, 0)
    crop_right = st.slider("Crop Right (%)", 0, 45, 0)
    crop_bottom = st.slider("Crop Bottom (%)", 0, 45, 0)
    rotate_deg = st.slider("Rotate (Degrees)", -180, 180, 0)

    st.markdown("---")

    save_name = st.text_input("Save as Filename:", "adjusted_latest.png")

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== APPLY ADJUSTMENTS =====================
result = img.copy()

w, h = result.size
left = int(w * (crop_left / 100.0))
top = int(h * (crop_top / 100.0))
right = w - int(w * (crop_right / 100.0))
bottom = h - int(h * (crop_bottom / 100.0))

if right > left and bottom > top:
    try:
        result = result.crop((left, top, right, bottom))
    except:
        result = img.copy()

result = result.resize((350, 350))

if rotate_deg != 0:
    result = result.rotate(rotate_deg, expand=True).resize((350, 350))

result = ImageEnhance.Brightness(result).enhance(brightness)
result = ImageEnhance.Contrast(result).enhance(contrast)
result = ImageEnhance.Sharpness(result).enhance(sharpness)

if blur_amt > 0:
    result = result.filter(ImageFilter.GaussianBlur(blur_amt))

if smooth_edges > 0:
    repeat = max(1, smooth_edges // 5)
    for _ in range(repeat):
        result = result.filter(ImageFilter.SMOOTH_MORE)

if opacity < 1.0:
    white_bg = Image.new("RGB", result.size, (255, 255, 255))
    result = Image.blend(result, white_bg, 1 - opacity)

# ===================== RIGHT PREVIEW =====================
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Preview")
    st.image(result, use_container_width=True)

    # Download
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Centered Download Button Only
    download_html = f'''
    <div class="btn-row" style="justify-content:center;">
        <a class="action-btn"
           href="data:image/png;base64,{b64}"
           download="{save_name}">
           Download
        </a>
    </div>
    '''
    st.markdown(download_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
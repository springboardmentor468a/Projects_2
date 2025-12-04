import streamlit as st
import os
from PIL import Image

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="About VisionExtract", layout="wide")

# --------------------------------------
# THEME SYNC
# --------------------------------------
STEEL = "#9CC3E5"
DARK = "#0B1220"

theme = st.session_state.get("theme", "light")
BG = DARK if theme == "dark" else STEEL
TEXT = "#FFFFFF" if theme == "dark" else "#000000"

# --------------------------------------
# GLOBAL STYLE
# --------------------------------------
st.markdown(f"""
<style>

[data-testid="stAppViewContainer"] {{
    background-color: {BG};
    color: {TEXT};
}}

.topmenu {{
  display:flex; gap:25px; justify-content:center;
  background: rgba(255,255,255,0.18);
  padding:14px; border-radius:12px;
  margin-bottom: 30px;
  backdrop-filter: blur(6px);
}}

.topmenu a {{
  font-size:18px; 
  font-weight:700; 
  color:{TEXT};
  text-decoration:none;
  padding:8px 16px;
  border-radius:8px;
  transition:0.25s;
}}

.topmenu a:hover {{
  background: rgba(255,255,255,0.28);
}}

.section-title {{
    font-size:30px;
    font-weight:800;
    text-align:center;
    margin-top:20px;
    color:{TEXT};
}}

.image-card {{
    background: rgba(255,255,255,0.25);
    padding:20px;
    border-radius:20px;
    text-align:center;
    box-shadow:0px 8px 20px rgba(0,0,0,0.25);
    backdrop-filter: blur(6px);
}}

.image-card img {{
    width:350px;
    height:350px;
    border-radius:14px;
    object-fit:cover;
}}

.pill-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap:18px;
    margin-top:20px;
}}

.pill {{
    background: rgba(255,255,255,0.45);
    padding:14px 20px;
    border-radius:50px;
    font-size:17px;
    color:{TEXT};
    font-weight:600;
    text-align:center;
    box-shadow:0 6px 14px rgba(0,0,0,0.25);
}}

.badge-container {{
    display:flex;
    flex-wrap:wrap;
    gap:14px;
    margin-top:20px;
    justify-content:center;
}}

.badge {{
    background: rgba(255,255,255,0.30);
    padding:12px 20px;
    border-radius:12px;
    font-size:16px;
    color:{TEXT};
    font-weight:700;
}}

.hero-box {{
    background: rgba(255,255,255,0.20);
    padding:30px;
    border-radius:20px;
    margin-top:30px;
    text-align:center;
    box-shadow:0 6px 18px rgba(0,0,0,0.30);
    backdrop-filter: blur(6px);
}}

.hero-text {{
    font-size:18px;
    line-height:1.6;
    color:{TEXT};
}}

</style>

<div class="topmenu">
    <a href="/" target="_self">App</a>
    <a href="/about" target="_self">About</a>
    <a href="/history" target="_self">History</a>
    <a href="/settings" target="_self">Settings</a>
</div>

""", unsafe_allow_html=True)

# --------------------------------------
# PAGE TITLE
# --------------------------------------
st.markdown(f"<h1 style='text-align:center; color:{TEXT};'>About VisionExtract</h1>", unsafe_allow_html=True)
st.write(" ")

# --------------------------------------
# BEFORE & AFTER SECTION
# --------------------------------------
st.markdown("<div class='section-title'>Before and After</div>", unsafe_allow_html=True)

orig_path = "assets/originalimage.png"
mask_path = "assets/maskimage.png"

col1, col2, col3 = st.columns([1,2,1])
with col2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.subheader("Original")
        if os.path.exists(orig_path):
            st.image(Image.open(orig_path).resize((350,350)))
        else:
            st.warning("originalimage.png missing from /assets")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.subheader("Extracted Output")
        if os.path.exists(mask_path):
            st.image(Image.open(mask_path).resize((350,350)))
        else:
            st.warning("maskimage.png missing from /assets")
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------
# KEY HIGHLIGHTS
# --------------------------------------
st.markdown("<div class='section-title'>Key Highlights</div>", unsafe_allow_html=True)

st.markdown("""
<div class="pill-grid">
    <div class="pill">Object Extraction with Deep Learning</div>
    <div class="pill">Multiple Background Modes</div>
    <div class="pill">Dark/Light Theme Support</div>
    <div class="pill">Glassmorphism UI Design</div>
    <div class="pill">History Storage for Outputs</div>
    <div class="pill">Real-Time Fast Processing</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------
# TECHNOLOGIES USED
# --------------------------------------
st.markdown("<div class='section-title'>Technologies Used</div>", unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <div class="badge">UNet + ResNet101</div>
    <div class="badge">PyTorch</div>
    <div class="badge">Streamlit</div>
    <div class="badge">NumPy</div>
    <div class="badge">PIL</div>
    <div class="badge">Custom CSS</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------
# PROJECT OVERVIEW
# --------------------------------------
st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class='hero-box'>
    <p class='hero-text'>
        VisionExtract is an AI-powered image segmentation tool designed to isolate the main object 
        from images with high accuracy. It provides multiple background customization options, 
        real-time processing, and a clean modern user interface.  
        The project demonstrates how deep learning can be integrated into a practical, user-friendly 
        system for design and visual editing workflows.
    </p>
</div>
""", unsafe_allow_html=True)

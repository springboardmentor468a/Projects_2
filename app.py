import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import os
import base64
import segmentation_models_pytorch as smp

# --------------------------------------
# APP CONFIG
# --------------------------------------
st.set_page_config(page_title="VisionExtract", layout="wide")

# --------------------------------------
# THEME
# --------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

dark_mode = st.toggle("Dark Mode", value=False)
st.session_state["theme"] = "dark" if dark_mode else "light"

STEEL = "#9CC3E5"
DARK = "#0B1220"

BG = DARK if st.session_state["theme"] == "dark" else STEEL
TEXT = "#FFFFFF" if st.session_state["theme"] == "dark" else "#000000"

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
  display:flex; gap:22px; justify-content:center;
  background: rgba(255,255,255,0.18);
  padding:12px; border-radius:12px;
  margin-bottom: 25px;
  margin-top: -40px;
}}
.topmenu a {{
  font-size:17px; font-weight:700; color:{TEXT};
  text-decoration:none; padding:8px 16px;
  border-radius:8px;
}}
.topmenu a:hover {{
  background: rgba(255,255,255,0.30);
}}

.download-btn {{
    background-color: #3498DB !important;
    color: #FFFFFF !important;
    padding: 14px 40px !important;
    border-radius: 50px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    text-decoration: none !important;
    display: inline-block !important;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.12) !important;
    transition: 0.25s ease-in-out !important;
    border: none !important;
}}
.download-btn:hover {{
    background-color: #1F78C8 !important;
    transform: translateY(-2px) !important;
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
# LOAD MODEL
# --------------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "final_unet_resnet101_rgb (1).pth"

    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )

    model.extra_head = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 1)
    )

    original_fwd = model.forward
    def new_fwd(x):
        return model.extra_head(original_fwd(x))
    model.forward = new_fwd

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


model, device = load_model()

# --------------------------------------
# FUNCTIONS
# --------------------------------------
def preprocess(img):
    img = img.resize((350, 350))
    arr = np.array(img) / 255.0
    arr = torch.tensor(arr, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    return arr.to(device)

def predict_mask(x):
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        return (pred > 0.5).float()

def apply_bg(mask, img, opt, custom=None):
    mask_np = mask.squeeze().cpu().numpy()
    mask3 = np.repeat(mask_np[...,None], 3, axis=2)
    img_np = np.array(img)/255

    if opt=="Black": bg=np.zeros_like(img_np)
    elif opt=="White": bg=np.ones_like(img_np)
    elif opt=="Steel Blue": bg=np.full_like(img_np,[127/255,167/255,201/255])
    elif opt=="Gradient":
        x=np.linspace(0,1,350)
        bg=np.stack([np.tile(x,(350,1))]*3,axis=2)
    elif opt=="Pattern":
        p=np.indices((350,350)).sum(0)%2
        bg=np.stack([p,p,p],axis=2)
    elif opt=="Custom Image" and custom is not None:
        bg=np.array(custom.resize((350,350)))/255
    else:
        bg=np.zeros_like(img_np)

    out=img_np*mask3 + bg*(1-mask3)
    return (out*255).astype("uint8")

# --------------------------------------
# TITLE
# --------------------------------------
st.markdown(f"<h1 style='text-align:center; color:{TEXT};'>VisionExtract</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:{TEXT}; font-size:20px;'>AI-based Object Extraction with Background Customization</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------
# SAMPLE EXAMPLE (CENTERED)
# --------------------------------------
st.markdown(f"<h3 style='text-align:center;'>Sample Example</h3>", unsafe_allow_html=True)

orig = Image.open("assets/image19.jpeg").resize((350,350))
maskd = Image.open("assets/extracted.png").resize((350,350))

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    colA, colB = st.columns(2)
    with colA:
        st.write("Original")
        st.image(orig)
    with colB:
        st.write("Extracted Output")
        st.image(maskd)

st.markdown("---")

# --------------------------------------
# BACKGROUND OPTIONS
# --------------------------------------
bg_opt = st.selectbox("Select Background", 
    ["Black", "White", "Steel Blue", "Gradient", "Pattern", "Custom Image"])

custom = None
if bg_opt=="Custom Image":
    up = st.file_uploader("Upload Background", type=["jpg","png","jpeg"])
    if up: custom = Image.open(up)

# --------------------------------------
# USER UPLOAD IMAGE
# --------------------------------------
uploaded = st.file_uploader("Upload your image", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((350,350))
    tensor = preprocess(img)
    mask = predict_mask(tensor)

    out_arr = apply_bg(mask, img, bg_opt, custom)
    out_img = Image.fromarray(out_arr)

    # CENTERED OUTPUTS
    c4, c5, c6 = st.columns([1, 2, 1])
    with c5:
        left, right = st.columns(2)

        with left:
            st.markdown(f"<h4 style='text-align:center; color:{TEXT};'>Original</h4>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)

        with right:
            st.markdown(f"<h4 style='text-align:center; color:{TEXT};'>Extracted Output</h4>", unsafe_allow_html=True)
            st.image(out_img, use_container_width=True)

    # SAVE history + latest
    os.makedirs("history", exist_ok=True)
    out_img.save("history/latest.png")

    count = len([f for f in os.listdir("history") if f.endswith(".png")])
    out_img.save(f"history/output_{count+1}.png")

    # CENTERED DOWNLOAD BUTTON
    # CENTERED DOWNLOAD BUTTON
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    st.markdown(
        """
        <div style="width:100%; display:flex; justify-content:center; margin-top:25px;">
            <a class="download-btn" href="data:image/png;base64,{b64}" download="extracted.png">
                Download Extracted Image
            </a>
        </div>
        """.format(b64=b64),
        unsafe_allow_html=True 
   )
    st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)


    st.success("Image Extracted Successfully!")

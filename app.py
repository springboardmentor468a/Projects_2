import streamlit as st
try:
    import torch
    import torch.nn as nn
    from torchvision import models
except Exception:
    torch = None
    nn = None
    models = None

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
from io import BytesIO
import requests

# --- 1. CONFIGURATION & VIBRANT CSS ---
st.set_page_config(
    page_title="NeonExtract",
    page_icon="🦄",
    layout="wide"
)

st.markdown("""
    <style>
    /* Minimal clean UI styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }

    /* Page background and container spacing */
    .css-1d391kg { padding-top: 18px; padding-bottom: 18px; }

    .main-title { color: #0f1724; font-size: 28px; font-weight: 700; margin: 0 0 6px 0; }
    .subtitle { color: #475569; margin: 0 0 18px 0; font-size: 14px; }

    /* Card style: light, subtle, consistent spacing */
    .card-box { background: #ffffff; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }

    /* Buttons: simple primary and secondary variants */
    div.stButton > button[kind="primary"] { background: #2563eb; color: white; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
    div.stButton > button[kind="primary"]:hover { filter: brightness(0.95); }
    div.stButton > button[kind="secondary"] { background: #eef2ff; color: #3730a3; border-radius: 8px; }
    div.stButton > button[kind="secondary"]:hover { background: #e0e7ff; }

    /* Images: rounded and consistent */
    .stImage > img { border-radius: 8px; }

    /* Form controls spacing */
    .stSlider, .stColorPicker, .stCheckbox { margin-bottom: 8px; }

    /* Reduce strong shadows for a cleaner look */
    .css-1vbd788 { box-shadow: none; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MODEL DEFINITION (Standard) ---
if nn is not None:
    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )
        def forward(self, x): return self.conv(x)

    class ResNetUNet(nn.Module):
        def __init__(self, n_classes=1):
            super().__init__()
            base = models.resnet34(weights=None) 
            self.base_layers = list(base.children())
            self.layer0 = nn.Sequential(*self.base_layers[:3]) 
            self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
            self.layer2 = self.base_layers[5] 
            self.layer3 = self.base_layers[6] 
            self.layer4 = self.base_layers[7] 
            self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.decoder4 = ConvBlock(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.decoder3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.decoder2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
            self.decoder1 = ConvBlock(128, 64)
            self.final_upsample = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.final = nn.Conv2d(32, n_classes, 1)

        def forward(self, x):
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            d4 = self.decoder4(torch.cat([self.up4(x4), x3], 1))
            d3 = self.decoder3(torch.cat([self.up3(d4), x2], 1))
            d2 = self.decoder2(torch.cat([self.up2(d3), x1], 1))
            d1 = self.decoder1(torch.cat([self.up1(d2), x0], 1))
            return self.final(self.final_upsample(d1))

    @st.cache_resource
    def load_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = ResNetUNet(n_classes=1).to(device)
            model_path = "best_model_v2 (1).pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                return None, None
            model.eval()
            return model, device
        except:
            return None, None
else:
    def load_model(): return None, None

# --- 3. UPDATED HELPER FUNCTIONS (Interactive) ---

# Updated to accept threshold
def process_image(model, device, image, threshold=0.5):
    if not model: return None
    original_np = np.array(image)
    # Resize for consistency
    transform = A.Compose([A.Resize(320, 320), A.Normalize(), ToTensorV2()])
    input_tensor = transform(image=original_np)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_tensor)
        # Apply threshold here based on user input
        mask = (torch.sigmoid(preds) > threshold).float().squeeze().cpu().numpy()
    # Resize mask back to original size
    mask_resized = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

# Updated to accept a background color
def composite_bg(image, mask, bg_color_hex, transparent=False):
    if mask is None: return image

    # 1. Create Transparent Image with alpha from mask
    img_rgba = image.convert("RGBA")
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    img_rgba.putalpha(mask_pil)

    # 2. If explicit transparent mode requested, return RGBA with alpha
    if transparent:
        return img_rgba

    # 3. Otherwise create a solid RGB background from the provided hex color
    # Accepts '#RRGGBB' or 'RRGGBB' formats
    hex_clean = bg_color_hex.lstrip('#')
    if len(hex_clean) == 6:
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
    elif len(hex_clean) == 3:
        r = int(hex_clean[0]*2, 16)
        g = int(hex_clean[1]*2, 16)
        b = int(hex_clean[2]*2, 16)
    else:
        # Fallback to black
        r, g, b = 0, 0, 0

    bg_solid = Image.new("RGBA", img_rgba.size, (r, g, b, 255))
    bg_solid.paste(img_rgba, (0, 0), img_rgba)
    return bg_solid.convert("RGB")

@st.cache_data
def get_sample():
    url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&w=400&q=80"
    try:
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        return None

# --- 4. VIBRANT UI LAYOUT ---

# A. Header
st.markdown('<h1 class="main-title">🦄 NeonExtract</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">The interactive, colorful background remover.</p>', unsafe_allow_html=True)

model, device = load_model()
if model:
     st.toast("✨ AI Model Loaded & Ready!", icon="🚀")

# B. Sample Section (Wrapped in custom CSS class)
st.markdown('<div class="card-box">', unsafe_allow_html=True)
st.markdown("### 🎨 See the Magic")

sample_img = get_sample()

# Pre-calculate sample output (using default 0.5 threshold)
if 'sample_out' not in st.session_state and sample_img and model:
    mask = process_image(model, device, sample_img, threshold=0.5)
    # Use transparent BG for sample
    st.session_state['sample_out'] = composite_bg(sample_img, mask, "#000000", transparent=True)

col1, col2 = st.columns(2)
with col1:
    st.caption("Original")
    if sample_img: 
        st.image(sample_img, use_container_width=True)
with col2:
    st.caption("Extracted")
    if 'sample_out' in st.session_state: 
        st.image(st.session_state['sample_out'], use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.write("") 

# C. THE "TRY IT" BUTTON
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

def activate_uploader():
    st.session_state.show_uploader = True
    st.toast("Workspace Opened! Let's create.", icon="🎉")

b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
with b_col2:
    st.button("🚀 Launch Creative Workspace", 
              key="try_it_btn", 
              type="primary", 
              use_container_width=True, 
              on_click=activate_uploader)

st.write("")

# D. INTERACTIVE USER WORKSPACE
if st.session_state.show_uploader:
    st.markdown("---")
    
    # Wrap workspace in card style
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Your Creative Lab")
    # Back button to close the workspace and return to the main view
    def deactivate_uploader():
        st.session_state.show_uploader = False
        try:
            st.toast("Workspace closed.", icon="◀️")
        except Exception:
            pass

    back_col, _ = st.columns([1, 9])
    with back_col:
        st.button("← Back", key="back_btn", on_click=deactivate_uploader)
    
    # 1. CONTROLS SECTION (New Interactivity!)
    c1, c2 = st.columns(2)
    with c1:
        # SLIDER: Allows user to tune the mask
        threshold = st.slider("🎚️ Edge Sensitivity (Threshold)", 0.1, 0.9, 0.5, 0.05, help="Lower = keep more details. Higher = cleaner, tighter edges.")
    with c2:
        # COLOR PICKER: Choose background (Streamlit color picker expects 3- or 6-digit hex)
        bg_color = st.color_picker("🎨 New Background Color", "#000000", help="Pick a solid color.")
        # Provide an explicit toggle for transparent output instead of relying on alpha suffix
        transparent_mode = st.checkbox("Make background transparent (PNG)", value=False)

    st.write("") # Spacer

    # 2. UPLOAD & PROCESS
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        user_img = Image.open(uploaded_file).convert("RGB")
        
        user_result = None
        if model:
            # Interactive spinner
            with st.spinner("🔮 AI is performing alchemy..."):
                # Pass the slider threshold value here!
                u_mask = process_image(model, device, user_img, threshold=threshold)
                # Pass the color picker value and transparency flag here!
                user_result = composite_bg(user_img, u_mask, bg_color, transparent=transparent_mode)

        # Results Display
        uc1, uc2 = st.columns(2)
        with uc1:
            st.caption("Your Input")
            st.image(user_img, use_container_width=True)
        
        with uc2:
            st.caption("Your Creation")
            if user_result:
                st.image(user_result, use_container_width=True)
        
        # E. DOWNLOAD BUTTON
        if user_result:
            st.write("")
            
            # Determine format based on background choice
            save_fmt = "PNG" if transparent_mode else "JPEG"
            mime_type = "image/png" if transparent_mode else "image/jpeg"

            buf = BytesIO()
            user_result.save(buf, format=save_fmt)
            byte_im = buf.getvalue()
            
            d_col1, d_col2, d_col3 = st.columns([1, 2, 1])
            with d_col2:
                st.download_button(
                    label=f"💾 Download {save_fmt}",
                    data=byte_im,
                    file_name=f"neon_extract.{save_fmt.lower()}",
                    mime=mime_type,
                    type="secondary",
                    use_container_width=True
                )
    else:
         st.info("👆 Drag and drop an image above to start tweaking!")

    st.markdown('</div>', unsafe_allow_html=True)
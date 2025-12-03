# streamlit_segme_precise_mask_only.py
import io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import segmentation

# ----------------- Config -----------------
FG_THRESHOLD = 0.6  # stricter foreground threshold
MORPH_KERNEL = 7
MORPH_ITER = 3
KEEP_LARGEST = True

# ----------------- Model loader -----------------
@st.cache_resource
def load_model(device="cpu"):
    model = segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
    model.eval()
    model.to(device)
    return model

# ----------------- Preprocessing -----------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
def image_to_tensor(img_pil: Image.Image, device="cpu"):
    return preprocess(img_pil).unsqueeze(0).to(device)

# ----------------- Mask Helpers -----------------
def keep_largest_component(mask_np):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if num_labels <= 1:
        return mask_np
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype('uint8') * 255

def refine_mask(prob_map, threshold=FG_THRESHOLD, kernel_size=MORPH_KERNEL, morph_iter=MORPH_ITER, keep_largest=True):
    # strict binarization
    bin_mask = (prob_map >= threshold).astype('uint8') * 255
    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    if keep_largest:
        cleaned = keep_largest_component(cleaned)
    return cleaned

def sharpen_image(img: np.ndarray):
    blur = cv2.GaussianBlur(img, (0,0), 3)
    sharpened = cv2.addWeighted(img, 1.7, blur, -0.7, 0)
    return np.clip(sharpened,0,255).astype('uint8')

# ----------------- Core function -----------------
def segment_object_only(img_pil: Image.Image, model, device='cpu', bg_color=(0,0,0)):
    tensor = image_to_tensor(img_pil, device=device)
    with torch.no_grad():
        out = model(tensor)
        logits = out['out'][0] if isinstance(out, dict) and 'out' in out else out[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    
    fg_prob = np.max(probs[1:], axis=0)  # most confident foreground

    # Resize mask
    img_w, img_h = img_pil.size
    fg_prob_resized = cv2.resize(fg_prob, (img_w,img_h), interpolation=cv2.INTER_NEAREST)

    # Refine mask strictly
    bin_mask = refine_mask(fg_prob_resized)

    # Composite only object
    src = np.array(img_pil).astype('float32')/255.0
    if src.ndim==2:
        src = np.stack([src]*3, axis=-1)
    bg = np.ones_like(src)*np.array(bg_color, dtype=np.float32)/255.0
    alpha_3 = np.stack([bin_mask/255]*3, axis=-1)
    out_rgb = src*alpha_3 + bg*(1-alpha_3)
    out_rgb = (out_rgb*255).astype('uint8')
    out_rgb = sharpen_image(out_rgb)

    return Image.fromarray(out_rgb)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="SegMe", layout="wide")
st.markdown("<h1 style='text-align:center; color:#1A5276;'>SegMe — Object-Only Masking</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Only the subject is visible; background completely removed.</p>", unsafe_allow_html=True)
st.markdown("---")

# Paths
BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"
sample_input_path = SAMPLES_DIR / "sample_input.jpg"
sample_output_path = SAMPLES_DIR / "sample_output.png"

# Sample Input/Output
st.subheader("Sample Input & Output")
col1, col2 = st.columns(2)
with col1:
    if sample_input_path.exists():
        sample_input_img = Image.open(sample_input_path).convert("RGB")
        st.image(sample_input_img, caption="Sample Input", use_container_width=False, width=300)
with col2:
    if sample_output_path.exists():
        sample_output_img = Image.open(sample_output_path).convert("RGB")
        st.image(sample_output_img, caption="Sample Output", use_container_width=False, width=300)

st.markdown("---")
st.subheader("Try it Yourself")
uploaded = st.file_uploader("Upload your image", type=['jpg','jpeg','png','webp'])
bg_color = st.color_picker("Background color", "#000000")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(device)

if uploaded is not None:
    user_img = Image.open(uploaded).convert("RGB")
    st.markdown("**User Input**")
    st.image(user_img, use_container_width=False, width=360)

    if st.button("Process Image"):
        with st.spinner("Generating precise object-only mask..."):
            out_img = segment_object_only(
                user_img, model, device=device,
                bg_color=tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
            )

        st.markdown("**User Output**")
        st.image(out_img, use_container_width=False, width=360)

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Masked PNG", data=buf, file_name="masked_output.png", mime="image/png")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps  # <--- Added ImageOps
import io
import requests

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="NeuroMask AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    h3 {
        color: #FAFAFA;
        text-align: center;
        font-weight: 300;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
    .img-caption {
        text-align: center;
        font-style: italic;
        color: #cccccc;
    }
    div[data-testid="stFileUploader"] {
        width: 60%;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING & FUNCTIONS
# ==========================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def combo_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

iou_metric = tf.keras.metrics.MeanIoU(num_classes=2, name="mean_io_u")

@st.cache_resource
def load_model():
    MODEL_PATH = 'final.keras' 
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'combo_loss': combo_loss,
                'dice_loss': dice_loss,
                'mean_io_u': iou_metric
            }
        )
        return model
    except Exception as e:
        st.error(f"Error loading model. Make sure '{MODEL_PATH}' is in the same folder.")
        st.stop()

model = load_model()

def process_image(pil_image):
    """
    Full processing pipeline:
    1. Fix Rotation
    2. Predict Mask (256x256)
    3. Resize Mask back to Original Size
    4. Apply Mask to Original Image
    """
    

    # This checks the hidden EXIF tag and rotates the image upright
    pil_image = ImageOps.exif_transpose(pil_image)
    
    # Save original size for later (Width, Height)
    org_size = pil_image.size 
    
    # --- 2. PREPARE FOR MODEL (256x256) ---
    img_resized = pil_image.resize((256, 256))
    img_array = np.array(img_resized)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_tensor = tf.cast(img_array, tf.float32)
    img_tensor = tf.keras.applications.resnet50.preprocess_input(img_tensor)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # --- 3. PREDICT ---
    prediction = model.predict(img_tensor, verbose=0)
    pred_mask = (prediction[0] > 0.5).astype(np.uint8)
    
    # --- 4. RESIZE MASK BACK TO ORIGINAL SIZE ---
    # Convert mask to PIL image to resize it easily
    # We multiply by 255 to make it a visible black/white image
    mask_pil = Image.fromarray(pred_mask[:, :, 0] * 255)
    
    # Resize mask to match the ORIGINAL image dimensions
    mask_pil = mask_pil.resize(org_size, resample=Image.NEAREST)
    
    # Convert back to numpy (0 or 255)
    mask_resized = np.array(mask_pil) // 255  # Back to 0 or 1
    
    # --- 5. APPLY MASK TO ORIGINAL IMAGE ---
    original_array = np.array(pil_image)
    
    # Ensure mask has 3 channels (R, G, B)
    mask_3d = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)
    
    # Apply mask: Keep original pixel where mask is 1, else Black
    segmented_array = np.where(mask_3d == 1, original_array, 0)
    
    # Return BOTH the fixed original and the result
    return pil_image, Image.fromarray(segmented_array.astype('uint8'))

# ==========================================
# 3. UI LAYOUT
# ==========================================

st.markdown("<h1>✨ NeuroMask AI ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3>Advanced Image Segmentation & Background Removal</h3>", unsafe_allow_html=True)
st.divider()

# --- Sample Section ---
st.markdown("#### 👁️ See what it can do")

# Use columns to center the sample
c1, c2, c3 = st.columns([1, 3, 1])

with c2:
    col1, col2 = st.columns(2)
    
    # Load Local Images
    # Make sure these files exist in your folder!
    try:
        sample_org = Image.open("sample_original.png")
        sample_seg = Image.open("sample_segmented.png")
        
        with col1:
            st.image(sample_org, caption="Original", use_container_width=True)
        with col2:
            st.image(sample_seg, caption="Segmented Output", use_container_width=True)
            
    except FileNotFoundError:
        # Fallback if files are missing
        st.warning("⚠️ Sample images not found. Please add 'sample_original.png' and 'sample_segmented.png' to your folder.")
        
st.divider()

# --- "Try It" Section ---
st.markdown("<h2 style='text-align: center;'>🚀 Try It Yourself</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    user_image = Image.open(uploaded_file).convert('RGB')
    
    st.write("---")
    
    with st.spinner('🤖 AI is processing...'):
        # Get BOTH the fixed original (rotated) and the result
        fixed_original, result_image = process_image(user_image)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # Display the ROTATION-FIXED original
        st.image(fixed_original, use_container_width=True)
    
    with col2:
        st.subheader("Predicted Output")
        # Display the High-Res Result
        st.image(result_image, use_container_width=True)

    st.write("---")
    
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        st.download_button(
            label="⬇️ Download Segmented Image",
            data=byte_im,
            file_name="segmented_output.png",
            mime="image/png"
        )

st.markdown("<br><br><p style='text-align:center; color:grey;'>Powered by TensorFlow & ResNet50 | Created with Streamlit</p>", unsafe_allow_html=True)
# Vision Extract - Real Mask Generation with Neon Theme (Fixed)
# Save this as: vision_extract_app.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import io
import zipfile
from tensorflow import keras
import tensorflow as tf

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Vision Extract",
    page_icon="🖌️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# NEON THEME CSS (From template) - FIXED
# ============================================================================
st.markdown("""
    <style>
    /* Main background - navy blue */
    .stApp {
        background: #0a1628;
    }
    
    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #00b4db 0%, #00ffcc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5rem !important;
        filter: drop-shadow(0 0 5px rgba(0, 255, 204, 0.5));
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    h2 {
        background: linear-gradient(135deg, #0083b0 0%, #00b4db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        filter: drop-shadow(0 0 2px rgba(0, 180, 219, 0.2));
    }
    
    h3 {
        background: linear-gradient(135deg, #00b4db 0%, #00d4a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    /* Regular text with gradient */
    .stMarkdown p, label, span {
        background: linear-gradient(135deg, #80deea 0%, #a7ffeb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Remove button border/background for header logo */
    button[kind="header"] {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    button[kind="header"]:hover {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
    }
    
    /* Button styling with neon borders */
    .stButton > button {
        background: #0d1b2a;
        color: #00ffcc;
        border: 2px solid #00b4db;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 180, 219, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.6);
        transform: translateY(-2px);
        color: #00ffcc;
    }
    
    /* Cards with neon borders */
    .custom-card {
        background: #0d1b2a;
        border: 2px solid #00b4db;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 0 15px rgba(0, 180, 219, 0.3);
        margin: 1rem 0;
    }
    
    /* Image container */
    .image-container {
        text-align: center;
        padding: 1rem;
        background: #0d1b2a;
        border: 1px solid rgba(0, 180, 219, 0.4);
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0, 180, 219, 0.2);
    }
    
    .image-label {
        background: linear-gradient(135deg, #00b4db 0%, #00ffcc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* File uploader styling - FIXED */
    [data-testid="stFileUploader"] {
        background: #0d1b2a;
        border: 2px solid #00b4db;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 0 8px rgba(0, 180, 219, 0.3);
    }
    
    [data-testid="stFileUploader"] label {
        background: linear-gradient(135deg, #00b4db 0%, #00ffcc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    [data-testid="stFileUploader"] section {
        border: 2px dashed #00b4db !important;
        background: rgba(0, 180, 219, 0.05);
    }
    
    [data-testid="stFileUploader"] section:hover {
        border-color: #00ffcc !important;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.3);
        background: rgba(0, 255, 204, 0.1);
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #00b4db 0%, #00ffcc 100%);
        color: #0a1628;
        border: none;
        font-weight: 600;
    }
    
    [data-testid="stFileUploader"] button:hover {
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);
    }
    
    [data-testid="stFileUploader"] span {
        color: #00ffcc !important;
        background: none !important;
        -webkit-text-fill-color: #00ffcc !important;
    }
    
    .uploadedFile {
        background: #0d1b2a;
        border: 1px solid #00b4db;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 0 8px rgba(0, 180, 219, 0.3);
    }
    
    /* Radio button styling - FIXED */
    .stRadio > div {
        background: #0d1b2a;
        border: 1px solid rgba(0, 180, 219, 0.3);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stRadio label {
        background: linear-gradient(135deg, #80deea 0%, #a7ffeb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Radio button circles */
    .stRadio input[type="radio"] {
        accent-color: #00ffcc;
    }
    
    .stRadio > div > label > div:first-child {
        background-color: #0d1b2a !important;
        border: 2px solid #00b4db !important;
    }
    
    .stRadio > div > label > div:first-child:hover {
        border-color: #00ffcc !important;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.4);
    }
    
    /* Checkbox styling - FIXED for visibility */
    .stCheckbox {
        background: #0d1b2a;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stCheckbox label {
        background: linear-gradient(135deg, #80deea 0%, #a7ffeb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    /* Checkbox itself - white checkmark */
    .stCheckbox input[type="checkbox"] {
        accent-color: #00ffcc;
        width: 20px;
        height: 20px;
    }
    
    /* Make checkbox border visible */
    .stCheckbox > label > div:first-child {
        background-color: #0d1b2a !important;
        border: 2px solid #00b4db !important;
        border-radius: 4px;
    }
    
    .stCheckbox > label > div:first-child:hover {
        border-color: #00ffcc !important;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.4);
    }
    
    /* Checked state */
    .stCheckbox input[type="checkbox"]:checked + div {
        background-color: #00ffcc !important;
        border-color: #00ffcc !important;
    }
    
    /* Info note */
    .info-note {
        background: #0d1b2a;
        border-left: 4px solid #00ffcc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
    }
    
    /* Select box */
    .stSelectbox > div > div > select {
        background-color: #0d1b2a !important;
        color: #00ffcc !important;
        border: 2px solid #00b4db !important;
        border-radius: 8px;
        box-shadow: 0 0 8px rgba(0, 180, 219, 0.3);
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: #00ffcc !important;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.5) !important;
    }
    
    .stSelectbox label {
        background: linear-gradient(135deg, #80deea 0%, #a7ffeb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Color picker */
    .stColorPicker label {
        background: linear-gradient(135deg, #80deea 0%, #a7ffeb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00b4db 0%, #00ffcc 100%);
        color: #0a1628;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 0 25px rgba(0, 255, 204, 0.7);
        transform: translateY(-2px);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #0d1b2a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00b4db 0%, #00ffcc 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    
    /* Remove extra containers/boxes */
    div[data-testid="stVerticalBlock"] > div:first-child {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'bg_type_single' not in st.session_state:
    st.session_state.bg_type_single = 'original'
if 'bg_type_multi' not in st.session_state:
    st.session_state.bg_type_multi = 'original'

# ============================================================================
# CUSTOM LAYER FOR MODEL
# ============================================================================
class CropAndConcat(keras.layers.Layer):
    def call(self, inputs):
        upsampled, encoder_features = inputs
        up_shape = tf.shape(upsampled)
        enc_shape = tf.shape(encoder_features)
        offset_h = (enc_shape[1] - up_shape[1]) // 2
        offset_w = (enc_shape[2] - up_shape[2]) // 2
        cropped = encoder_features[:, 
                                    offset_h:offset_h + up_shape[1],
                                    offset_w:offset_w + up_shape[2],
                                    :]
        return tf.concat([upsampled, cropped], axis=-1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained U-Net model"""
    try:
        custom_objects = {
            'CropAndConcat': CropAndConcat,
            'dice_coefficient': lambda y_true, y_pred: tf.constant(0.0)
        }
        model = keras.models.load_model('best_model.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(250, 250)):
    """Preprocess image for model"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized

def apply_background(masked_image_array, background_type, color1=None, color2=None):
    """Apply background to masked image"""
    height, width = masked_image_array.shape[:2]
    
    # Create alpha channel from mask (non-zero pixels)
    if len(masked_image_array.shape) == 3:
        mask = np.any(masked_image_array > 0, axis=2).astype(np.uint8) * 255
    else:
        mask = (masked_image_array > 0).astype(np.uint8) * 255
    
    if background_type == 'transparent':
        # Convert to RGBA
        if len(masked_image_array.shape) == 2:
            rgba = cv2.cvtColor(masked_image_array, cv2.COLOR_GRAY2RGBA)
        elif masked_image_array.shape[2] == 3:
            rgba = cv2.cvtColor(masked_image_array, cv2.COLOR_RGB2RGBA)
        else:
            rgba = masked_image_array.copy()
        rgba[:, :, 3] = mask
        return Image.fromarray(rgba)
    
    elif background_type == 'solid':
        # Create solid color background
        bg = np.full((height, width, 3), color1, dtype=np.uint8)
        mask_3channel = mask[:, :, np.newaxis] / 255.0
        result = (masked_image_array * mask_3channel + bg * (1 - mask_3channel)).astype(np.uint8)
        return Image.fromarray(result)
    
    elif background_type == 'gradient':
        # Create gradient background
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            ratio = i / height
            bg[i, :] = [
                int(color1[0] * (1 - ratio) + color2[0] * ratio),
                int(color1[1] * (1 - ratio) + color2[1] * ratio),
                int(color1[2] * (1 - ratio) + color2[2] * ratio)
            ]
        mask_3channel = mask[:, :, np.newaxis] / 255.0
        result = (masked_image_array * mask_3channel + bg * (1 - mask_3channel)).astype(np.uint8)
        return Image.fromarray(result)
    
    return Image.fromarray(masked_image_array)

def generate_real_mask(model, image, original_format='PNG'):
    """Generate REAL mask (not predicted) - extracts actual objects"""
    original_width, original_height = image.size
    
    # Preprocess
    processed = preprocess_image(image, target_size=(250, 250))
    input_batch = np.expand_dims(processed, axis=0)
    
    # Predict mask
    prediction = model.predict(input_batch, verbose=0)
    mask = prediction[0, :, :, 0]
    
    # Threshold to get binary mask
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # Apply mask to original image to extract REAL masked region
    masked_region = processed * binary_mask[:, :, np.newaxis]
    masked_region_uint8 = (masked_region * 255).astype(np.uint8)
    
    # Resize back to original dimensions
    masked_pil = Image.fromarray(masked_region_uint8)
    masked_pil_resized = masked_pil.resize((original_width, original_height), Image.LANCZOS)
    
    return masked_pil_resized, original_format

def resize_for_display(image, max_width=300, max_height=200):
    """Resize image for display while maintaining aspect ratio"""
    width, height = image.size
    ratio = min(max_width/width, max_height/height)
    new_size = (int(width * ratio), int(height * ratio))
    return image.resize(new_size, Image.LANCZOS)

def create_download_zip(results_list):
    """Create ZIP file with all images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, (original, masked, file_format) in enumerate(results_list, 1):
            img_buffer = io.BytesIO()
            masked.save(img_buffer, format=file_format)
            ext = file_format.lower()
            if ext == 'jpeg':
                ext = 'jpg'
            zip_file.writestr(f'masked_image_{idx}.{ext}', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

# ============================================================================
# HEADER COMPONENT - FIXED (No box, larger, bold)
# ============================================================================
def render_header():
    """Render clickable header without box"""
    if st.button("🖌️ Vision Extract", key=f"header_btn_{st.session_state.page}", type="secondary"):
        st.session_state.page = 'home'
        st.session_state.uploaded_files = []
        st.session_state.results = []
        st.rerun()

# ============================================================================
# PAGE 1: HOME
# ============================================================================
def home_page():
    st.markdown("<h1>🖌️ Vision Extract</h1>", unsafe_allow_html=True)
    
    #st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        #st.markdown('<div class="image-container">', unsafe_allow_html=True)
        try:
            sample_img = Image.open('sample_input.png')
            display_img = resize_for_display(sample_img, 300, 200)
            st.image(display_img, use_container_width=True)
            st.markdown('<p class="image-label">Sample Image</p>', unsafe_allow_html=True)
        except:
            st.info("📸 Place 'sample_input.png' in the same folder")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        #st.markdown('<div class="image-container">', unsafe_allow_html=True)
        try:
            sample_output = Image.open('sample_output.png')
            display_output = resize_for_display(sample_output, 300, 200)
            st.image(display_output, use_container_width=True)
            st.markdown('<p class="image-label">Sample Masked Output</p>', unsafe_allow_html=True)
        except:
            st.info("📸 Place 'sample_output.png' in the same folder")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("START", use_container_width=True, type="primary"):
            st.session_state.page = 'upload'
            st.rerun()

# ============================================================================
# PAGE 2: UPLOAD - FIXED (Radio instead of checkbox)
# ============================================================================
def upload_page():
    st.markdown("<h1>🖌️ Vision Extract</h1>", unsafe_allow_html=True)
    
    #st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    if st.session_state.uploaded_files:
        st.markdown("### 📁 Uploaded Images")
        for file in st.session_state.uploaded_files:
            st.markdown(f"✓ {file.name}")
    else:
        st.markdown("### 📸 Add Images")
        st.info("No images uploaded yet")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # FIXED: Radio button for single selection
    upload_mode = st.radio(
        "Select upload mode",
        ["Single Image", "Multiple Images"],
        key="upload_mode",
        horizontal=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if upload_mode == "Single Image":
        uploaded = st.file_uploader(
            "➕ Add Files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=False,
            key="single_uploader"
        )
        if uploaded:
            st.session_state.uploaded_files = [uploaded]
    else:
        uploaded = st.file_uploader(
            "➕ Add Files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="multi_uploader"
        )
        if uploaded:
            st.session_state.uploaded_files = uploaded
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generate", use_container_width=True, type="primary", disabled=not st.session_state.uploaded_files):
        with st.spinner("🔄 Processing images..."):
            if st.session_state.model is None:
                st.session_state.model = load_model()
            
            if st.session_state.model is None:
                st.error("Failed to load model. Ensure 'best_model.h5' is in the folder.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            results = []
            for uploaded_file in st.session_state.uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1].upper()
                if file_extension == 'JPG':
                    file_extension = 'JPEG'
                
                original_img = Image.open(uploaded_file)
                masked_img, img_format = generate_real_mask(st.session_state.model, original_img, file_extension)
                results.append((original_img, masked_img, img_format))
            
            st.session_state.results = results
            
            if len(results) == 1:
                st.session_state.page = 'result_single'
            else:
                st.session_state.page = 'result_multiple'
            st.rerun()
    
    st.markdown('<div class="info-note">', unsafe_allow_html=True)
    st.markdown("**Note:** Upload only .jpg, .jpeg, .png format files")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 3: SINGLE RESULT - FIXED (unique keys)
# ============================================================================
def result_single_page():
    st.markdown("<h1>🖌️ Vision Extract</h1>", unsafe_allow_html=True)
    
    #st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    if st.session_state.results:
        original_img, masked_img, img_format = st.session_state.results[0]
        orig_width, orig_height = original_img.size
        
        col1, col2 = st.columns(2)
        
        with col1:
            #st.markdown('<div class="image-container">', unsafe_allow_html=True)
            display_orig = resize_for_display(original_img, 300, 200)
            st.image(display_orig, caption=f"Original ({orig_width}×{orig_height})", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            #st.markdown('<div class="image-container">', unsafe_allow_html=True)
            display_masked = resize_for_display(masked_img, 300, 200)
            st.image(display_masked, caption=f"Masked ({orig_width}×{orig_height})", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Background options
        st.markdown("### 🎨 Background Options")
        bg_type = st.selectbox(
            "Select background type",
            ["original", "transparent", "solid", "gradient"],
            key="bg_select_single"
        )
        
        final_img = masked_img.copy()
        
        if bg_type == "solid":
            color = st.color_picker("Pick a color", "#FFFFFF", key="solid_color_single")
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            masked_array = np.array(masked_img)
            final_img = apply_background(masked_array, 'solid', rgb)
        
        elif bg_type == "gradient":
            col1, col2 = st.columns(2)
            with col1:
                color1 = st.color_picker("Start color", "#FF0000", key="grad_start_single")
            with col2:
                color2 = st.color_picker("End color", "#0000FF", key="grad_end_single")
            rgb1 = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
            rgb2 = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
            masked_array = np.array(masked_img)
            final_img = apply_background(masked_array, 'gradient', rgb1, rgb2)
        
        elif bg_type == "transparent":
            masked_array = np.array(masked_img)
            final_img = apply_background(masked_array, 'transparent')
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download button
        img_buffer = io.BytesIO()
        save_format = 'PNG' if bg_type == 'transparent' else img_format
        final_img.save(img_buffer, format=save_format)
        img_buffer.seek(0)
        
        ext = save_format.lower()
        if ext == 'jpeg':
            ext = 'jpg'
        
        st.download_button(
            label=f"⬇️ Download Image (.{ext})",
            data=img_buffer,
            file_name=f"masked_image.{ext}",
            mime=f"image/{save_format.lower()}",
            use_container_width=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔄 Start Again", use_container_width=True):
            st.session_state.page = 'home'
            st.session_state.uploaded_files = []
            st.session_state.results = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 4: MULTIPLE RESULTS - FIXED (unique keys)
# ============================================================================
def result_multiple_page():
    st.markdown("<h1>🖌️ Vision Extract</h1>", unsafe_allow_html=True)
    
    #st.markdown('<div class="custom-card" style="max-height: 70vh; overflow-y: auto;">', unsafe_allow_html=True)
    
    if st.session_state.results:
        for idx, (original_img, masked_img, img_format) in enumerate(st.session_state.results, 1):
            orig_width, orig_height = original_img.size
            
            st.markdown(f"### Image {idx} ({orig_width}×{orig_height}, {img_format})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                display_orig = resize_for_display(original_img, 300, 200)
                st.image(display_orig, caption="Original", use_container_width=True)
            
            with col2:
                display_masked = resize_for_display(masked_img, 300, 200)
                st.image(display_masked, caption="Masked", use_container_width=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Background options for all - FIXED with unique keys
    st.markdown("### 🎨 Background Options (Applied to All)")
    bg_type = st.selectbox(
        "Select background type",
        ["original", "transparent", "solid", "gradient"],
        key="bg_select_multi_unique"
    )
    
    processed_results = []
    color_rgb = None
    color1_rgb = None
    color2_rgb = None
    
    if bg_type == "solid":
        color = st.color_picker("Pick a color", "#FFFFFF", key="solid_color_multi_unique")
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    
    elif bg_type == "gradient":
        col1, col2 = st.columns(2)
        with col1:
            color1 = st.color_picker("Start color", "#FF0000", key="grad_start_multi_unique")
            color1_rgb = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
        with col2:
            color2 = st.color_picker("End color", "#0000FF", key="grad_end_multi_unique")
            color2_rgb = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
    
    for original_img, masked_img, img_format in st.session_state.results:
        final_img = masked_img.copy()
        
        if bg_type == "solid" and color_rgb:
            masked_array = np.array(masked_img)
            final_img = apply_background(masked_array, 'solid', color_rgb)
        
        elif bg_type == "gradient" and color1_rgb and color2_rgb:
            masked_array = np.array(masked_img)
            final_img = apply_background(masked_array, 'gradient', color1_rgb, color2_rgb)
        
        elif bg_type == "transparent":
            masked_array = np.array(masked_img)
            final_img = apply_background(masked_array, 'transparent')
        
        save_format = 'PNG' if bg_type == 'transparent' else img_format
        processed_results.append((original_img, final_img, save_format))
    
    # Download all
    if processed_results:
        zip_buffer = create_download_zip(processed_results)
        
        st.download_button(
            label="⬇️ Download All Images (ZIP)",
            data=zip_buffer,
            file_name="masked_images.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔄 Start Again", use_container_width=True):
        st.session_state.page = 'home'
        st.session_state.uploaded_files = []
        st.session_state.results = []
        st.rerun()

# ============================================================================
# MAIN ROUTER
# ============================================================================
def main():
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'upload':
        upload_page()
    elif st.session_state.page == 'result_single':
        result_single_page()
    elif st.session_state.page == 'result_multiple':
        result_multiple_page()

if __name__ == "__main__":
    main()# Vision Extract - Real Mask Generation with Neon Theme
# Save this as: vision_extract_app.py


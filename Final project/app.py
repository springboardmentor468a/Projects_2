import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
import os
import tempfile
os.environ['NUMBA_CACHE_DIR'] = tempfile.gettempdir()
from rembg import remove
import zipfile

# ---------------------------------------------------------
# PAGE CONFIG + STYLE
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Segmentation", layout="wide", page_icon="✂️")

st.markdown("""
<style>
    .main {background-color: #0e1117; padding: 1rem;}
    .title {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        text-align: center;
        background: linear-gradient(90deg, #ff6ec4, #7873f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle {text-align: center; color: #ccc; font-size: 1.2rem; margin-bottom: 2rem;}

    /* Glow cards */
    .feature-card {
        padding: 18px;
        background: #1a1f2e;
        border-radius: 18px;
        border: 1px solid #333;
        box-shadow: 0px 0px 12px #161b25;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #ff6ec4, #7873f5) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 40px !important;
        font-weight: bold;
    }

    /* Hover zoom */
    img:hover {
        transform: scale(1.02);
        transition: 0.3s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Smart Segmentation</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered object isolation • Premium effects • Enhanced editing tools</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# ORIGINAL SAMPLE DISPLAY (YOUR EXISTING CODE)
# ---------------------------------------------------------

st.markdown("### Sample Result")
col1, col2 = st.columns(2)

try:
    sample_original = Image.open("stranger-things-eleven-teenage-girl-bzkmy8tmhyxxrcqq.jpg")

    # Create sample masked image
    input_bytes = io.BytesIO()
    sample_original.save(input_bytes, format="PNG")

    output = remove(
        input_bytes.getvalue(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=12
    )

    result = Image.open(io.BytesIO(output)).convert("RGBA")
    black_bg = Image.new("RGB", result.size, (0, 0, 0))
    sample_masked = Image.composite(result.convert("RGB"), black_bg, result.split()[-1])

    with col1:
        st.markdown("**Original Image**")
        st.image(sample_original, use_container_width=True)

    with col2:
        st.markdown("**Masked (Objects Highlighted)**")
        st.image(sample_masked, use_container_width=True)

except Exception as e:
    st.error(f"Sample load failed: {e}")

st.markdown("---")

# ---------------------------------------------------------
# USER UPLOAD SECTION (YOUR ORIGINAL CODE)
# ---------------------------------------------------------

st.markdown("### Upload Your Image")

uploaded_file = st.file_uploader(
    "Drag and Drop or Browse",
    type=["png", "jpg", "jpeg"],
    help="Upload any portrait, object, or animal image."
)

# ---------------------------------------------------------
# EXTRA ENHANCEMENTS + NEW FEATURES
# ---------------------------------------------------------

if uploaded_file:
    user_original = Image.open(uploaded_file)

    st.markdown("### Advanced Segmentation Settings")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        fg_threshold = st.slider("Foreground Threshold", 0, 255, 240)
    with col_b:
        bg_threshold = st.slider("Background Threshold", 0, 255, 15)
    with col_c:
        erode_size = st.slider("Edge Smoothness", 0, 20, 12)
    with col_d:
        bg_color = st.color_picker("Background Color", "#000000")

    st.markdown("---")
    st.markdown("### **Image Enhancement Tools**")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)

    with col_e1:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
    with col_e2:
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
    with col_e3:
        sharpness = st.slider("Sharpness", 0.5, 3.0, 1.0)
    with col_e4:
        auto_fix = st.checkbox("Auto Enhance")

    # SEGMENTATION STYLE
    st.markdown("### Segmentation Style")

    seg_style = st.selectbox(
        "Choose Style",
        ["Clean Cut", "Soft Glow", "Blur Background", "Hard Edges"],
    )

    # ---------------------------------------------------------
    # PROCESS IMAGE
    # ---------------------------------------------------------

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Your Original Image**")
        st.image(user_original, use_container_width=True)

    with col_right:
        st.markdown("**Your Enhanced Masked Result**")

        with st.spinner("Processing..."):
            # Remove background
            bytes_io = io.BytesIO()
            user_original.save(bytes_io, format="PNG")

            output = remove(
                bytes_io.getvalue(),
                alpha_matting=True,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
                alpha_matting_erode_size=erode_size
            )

            result = Image.open(io.BytesIO(output)).convert("RGBA")

            # BACKGROUND
            bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
            final_bg = Image.new("RGB", result.size, bg_rgb)
            masked = Image.composite(result.convert("RGB"), final_bg, result.split()[-1])

            # ENHANCEMENTS
            if auto_fix:
                enhancer = ImageEnhance.Color(masked)
                masked = enhancer.enhance(1.2)

            masked = ImageEnhance.Brightness(masked).enhance(brightness)
            masked = ImageEnhance.Contrast(masked).enhance(contrast)
            masked = ImageEnhance.Sharpness(masked).enhance(sharpness)

            # STYLE EFFECTS
            if seg_style == "Soft Glow":
                masked = masked.filter(ImageFilter.GaussianBlur(1.2))

            if seg_style == "Blur Background":
                blurred_bg = final_bg.filter(ImageFilter.GaussianBlur(12))
                masked = Image.composite(result.convert("RGB"), blurred_bg, result.split()[-1])

            if seg_style == "Hard Edges":
                masked = masked.filter(ImageFilter.EDGE_ENHANCE_MORE)

        st.image(masked, use_container_width=True)

        # ---------------------------------------------------------
        # DOWNLOAD SECTION
        # ---------------------------------------------------------

        buf_png = io.BytesIO()
        masked.save(buf_png, format="PNG")

        buf_jpg = io.BytesIO()
        masked.convert("RGB").save(buf_jpg, format="JPEG")

        st.download_button("Download PNG", buf_png.getvalue(),
                           "segmented_output.png", "image/png")

        st.download_button("Download JPG", buf_jpg.getvalue(),
                           "segmented_output.jpg", "image/jpeg")

        # ZIP Download
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zipf:
            zipf.writestr("output.png", buf_png.getvalue())
            zipf.writestr("output.jpg", buf_jpg.getvalue())

        st.download_button("Download ZIP (PNG + JPG)", zip_buf.getvalue(),
                           "segmented_images.zip")

else:
    st.info("Upload an image to start segmentation.")

st.markdown("<br><hr><center>Made with ❤️ • Smart Segmentation 2025</center>", unsafe_allow_html=True)

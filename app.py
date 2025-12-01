import streamlit as st
from PIL import Image, ImageOps
import io
from rembg import remove
import requests

# ------------------ Page Config & Style ------------------
st.set_page_config(page_title="Smart Segmentation", layout="centered", page_icon="Scissors")

st.markdown("""
<style>
    .main {background-color: #0e1117; padding: 2rem;}
    .title {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        text-align: center;
        background: linear-gradient(90deg, #ff6ec4, #7873f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {text-align: center; color: #bbb; font-size: 1.3rem; margin-bottom: 3rem;}
    .sample-box {background: #16213e; padding: 20px; border-radius: 15px; text-align: center;}
    .stButton>button {
        background: linear-gradient(90deg, #ff6ec4, #7873f5) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Smart Segmentation</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered object extraction • Perfect black background masks</p>', unsafe_allow_html=True)

# ------------------ Sample Images (Top Row) ------------------
st.markdown("### Sample Result")
col1, col2 = st.columns(2)

# Load sample original (local screenshot image)
try:
    sample_original = Image.open("Screenshot 2025-11-30 105059.png")

    # Generate sample masked version
    with st.spinner("Generating sample mask..."):
        input_bytes = io.BytesIO()
        sample_original.save(input_bytes, format="PNG")
        output = remove(input_bytes.getvalue(),
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=15,
                        alpha_matting_erode_size=12)
        result = Image.open(io.BytesIO(output)).convert("RGBA")
        black_bg = Image.new("RGB", result.size, (0, 0, 0))
        sample_masked = Image.composite(result.convert("RGB"), black_bg, result.split()[-1])

    with col1:
        st.markdown("**Original Image**")
        st.image(sample_original, use_column_width=True)
    with col2:
        st.markdown("**Masked (Objects Highlighted)**")
        st.image(sample_masked, use_column_width=True)
except Exception as e:
    st.error(f"Failed to load sample image: {e}. Please upload your own image below.")

st.markdown("---")

# ------------------ User Upload (Bottom Row) ------------------
st.markdown("### Try Your Own Image")
uploaded_file = st.file_uploader("Upload any photo (person, cat, dog, etc.)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    user_original = Image.open(uploaded_file)

    # Advanced Settings
    st.markdown("### Advanced Settings")
    col_settings1, col_settings2, col_settings3 = st.columns(3)
    with col_settings1:
        fg_threshold = st.slider("Foreground Threshold", 0, 255, 240, help="Higher values make foreground detection more strict")
    with col_settings2:
        bg_threshold = st.slider("Background Threshold", 0, 255, 15, help="Lower values make background detection more strict")
    with col_settings3:
        erode_size = st.slider("Erode Size", 0, 20, 12, help="Higher values smooth the edges more")

    bg_color = st.color_picker("Background Color", "#000000", help="Choose the background color for the masked image")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Your Original Image**")
        st.image(user_original, use_column_width=True)

    with col4:
        st.markdown("**Your Masked Result**")
        with st.spinner("Removing background..."):
            input_bytes = io.BytesIO()
            user_original.save(input_bytes, format="PNG")
            output = remove(input_bytes.getvalue(),
                            alpha_matting=True,
                            alpha_matting_foreground_threshold=fg_threshold,
                            alpha_matting_background_threshold=bg_threshold,
                            alpha_matting_erode_size=erode_size)
            result = Image.open(io.BytesIO(output)).convert("RGBA")
            # Convert hex color to RGB
            bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
            custom_bg = Image.new("RGB", result.size, bg_rgb)
            user_masked = Image.composite(result.convert("RGB"), custom_bg, result.split()[-1])

        st.image(user_masked, use_column_width=True)

        # Download button
        buf = io.BytesIO()
        user_masked.save(buf, format="PNG")
        st.download_button(
            label="Download Masked Image",
            data=buf.getvalue(),
            file_name="smart_segmentation_result.png",
            mime="image/png"
        )
else:
    st.info("Upload an image above to see your result!")

# Footer
st.markdown("<br><hr><center>Made with ❤️ • Smart Segmentation 2025</center>", unsafe_allow_html=True)
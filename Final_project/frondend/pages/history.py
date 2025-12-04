import streamlit as st
import os
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="History", layout="wide")

# ----------------------------------------------------
# THEME SYNC
# ----------------------------------------------------
STEEL = "#9CC3E5"
DARK = "#0B1220"
theme = st.session_state.get("theme", "light")
BG = DARK if theme == "dark" else STEEL
TEXT = "#FFFFFF" if theme == "dark" else "#000000"

# ----------------------------------------------------
# GLOBAL CSS (styles Streamlit-native buttons)
# - styles target .stButton and .stDownloadButton wrappers
# ----------------------------------------------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {BG};
    color: {TEXT};
}}

/* Top navbar look */
.topmenu {{
  display:flex; justify-content:center; gap:25px;
  background: rgba(255,255,255,0.18); padding:14px;
  border-radius:12px; backdrop-filter: blur(6px);
  margin-bottom: 25px;
}}
.topmenu a {{
  font-size:18px; font-weight:700; color:{TEXT}; text-decoration:none !important;
  padding:8px 16px; border-radius:8px;
}}
.topmenu a:hover {{ background: rgba(255,255,255,0.28); }}

/* Card look */
.card {{
  background: rgba(255,255,255,0.22);
  padding: 16px;
  border-radius: 18px;
  text-align:center;
  box-shadow:0 6px 18px rgba(0,0,0,0.25);
  backdrop-filter: blur(8px);
  margin-bottom: 25px;
}}

.filename {{
  margin-top: 12px;
  font-size:15px;
  font-weight:600;
  color:{TEXT};
}}

/* Buttons row wrapper */
.btn-row {{
  display:flex;
  justify-content:center;
  gap:18px;
  margin-top: 12px;
  align-items:center;
}}

/* Style for Streamlit download button wrapper */
.stDownloadButton > button {{
  background-color: #3498DB !important;
  color: #FFFFFF !important;
  padding: 8px 22px !important;
  border-radius: 25px !important;
  font-weight:700 !important;
  border: none !important;
  outline: none !important;
  text-decoration: none !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}}
.stDownloadButton > button:hover {{
  background-color: #1F78C8 !important;
  transform: translateY(-2px);
}}

/* Style for Streamlit normal button wrapper (Delete) */
.stButton > button {{
  background-color: #E74C3C !important;
  color: #FFFFFF !important;
  padding: 8px 22px !important;
  border-radius: 25px !important;
  font-weight:700 !important;
  border: none !important;      /* removes default border */
  outline: none !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}}
.stButton > button:hover {{
  background-color: #C0392B !important;
  transform: translateY(-2px);
}}

/* Make sure links in navbar have no underline */
.topmenu a {{ text-decoration: none !important; }}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# NAVBAR (keeps same links)
# ----------------------------------------------------
st.markdown("""
<div class="topmenu">
  <a href="/" target="_self">App</a>
  <a href="/about" target="_self">About</a>
  <a href="/history" target="_self">History</a>
  <a href="/settings" target="_self">Settings</a>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------
st.markdown(f"<h1 style='text-align:center; color:{TEXT};'>History</h1>", unsafe_allow_html=True)
st.write(" ")

# ----------------------------------------------------
# HISTORY FOLDER
# ----------------------------------------------------
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

files = sorted(os.listdir(HISTORY_DIR), reverse=True)
valid_ext = (".png", ".jpg", ".jpeg")
image_files = [f for f in files if f.lower().endswith(valid_ext)]

if len(image_files) == 0:
    st.info("No images saved in history yet.")
    st.stop()

# ----------------------------------------------------
# Show images in 3-column grid
# For each image: show image, filename, download (st.download_button), delete (st.button)
# ----------------------------------------------------
cols = st.columns(3)
idx = 0

for fname in image_files:
    path = os.path.join(HISTORY_DIR, fname)

    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        continue

    with cols[idx % 3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Display image (keeps aspect and uses container width)
        st.image(img, use_container_width=True)

        # Filename
        st.markdown(f"<div class='filename'>{fname}</div>", unsafe_allow_html=True)

        # Prepare image bytes for download
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_data = buf.getvalue()

        # Buttons: use two small columns to align side-by-side reliably
        col_dl, col_del = st.columns([1,1])

        with col_dl:
            # st.download_button is native and works reliably
            st.download_button(
                label="Download",
                data=byte_data,
                file_name=fname,
                mime="image/png",
                key=f"dl_{fname}"
            )
        with col_del:
            # Delete button (native). When clicked, remove file and rerun.
            if st.button("Delete", key=f"del_{fname}"):
                try:
                    os.remove(path)
                except Exception as e:
                    st.error(f"Could not delete: {e}")
                else:
                    st.success(f"{fname} deleted.")
                    st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    idx += 1

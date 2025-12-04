# 🪄 Smart Segmentation
### Advanced AI Background Removal, Replacement & Blur

> Smart Segmentation is a modern web application that leverages machine learning (U-Net) to automatically remove, blur, or replace image backgrounds with professional precision.

---

## 🚀 Live Demo
👉 **[Launch App on Streamlit Cloud](https://appapp-bps5hrhwbwpkqhantqba5u.streamlit.app)**

---

## ✨ Key Features

### 🧠 Core AI Capabilities
* **✂️ Precision Background Removal:** Powered by the robust `rembg` library (U-Net architecture) to accurately isolate subjects from complex backgrounds.
* **🌫️ DSLR-Style Blur:** Apply adjustable Gaussian blur to the background, simulating a high-end camera's depth of field (Bokeh effect).
* **🎨 Dynamic Replacement:** Instantly swap backgrounds with solid colors (HEX codes) or upload your own custom scenic images.

### 💻 Modern User Experience
* **📱 Responsive UI:** Fully optimized for Desktop, Tablet, and Mobile workflows using Streamlit's fluid layout.
* **🎥 Cinematic Design:** Features a sleek "Glassmorphism" design language and a cinematic hero section.
* **🖼️ Sample Gallery:** Includes a built-in gallery to test the model capabilities immediately without needing your own files.
* **📦 Batch Processing:** Drag and drop multiple images to process a queue of photos in seconds.

---


## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python-based UI framework)
* **Image Processing:** Pillow (PIL), NumPy, OpenCV
* **AI/ML Engine:** PyTorch, Rembg (v2.0.67)
* **Deployment:** Streamlit Cloud / Docker

---

## 📂 Project Structure

```bash
Smart-Segmentation/
├── app.py                 # Main Streamlit Application
├── requirements.txt       # Python dependencies
├── packages.txt           # System-level dependencies (for Linux)
├── .streamlit/
│   └── config.toml        # UI Theme configuration
├── static/                # Static assets (images, css)
└── README.md              # Project Documentation

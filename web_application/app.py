import os
import io
import zipfile
import base64
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from PIL import Image

# Import background operations (model loads internally ONE TIME)
from modules.background_ops import (
    remove_background,
    replace_background_color,
    replace_background_image,
    blur_background
)

app = Flask(__name__)
app.secret_key = "super_secret_key"

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"
BATCH_DIR = "static/batch_outputs"
SAMPLES_DIR = "static/samples"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)


# Utility: Load image → base64 for templates
def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ----------------------------------------
# INDEX PAGE
# ----------------------------------------
@app.route("/")
def index():
    return render_template("index.html", title="AIvision Extract")


# ----------------------------------------
# BACKGROUND REMOVAL
# ----------------------------------------
@app.route("/background-removal", methods=["GET", "POST"])
def background_removal():
    if request.method == "POST":

        if "image" not in request.files:
            flash("Please upload an image.", "error")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)

        input_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(input_path)

        # Process image
        output_bytes = remove_background(input_path)

        output_path = os.path.join(OUTPUT_DIR, "removed.png")
        with open(output_path, "wb") as f:
            f.write(output_bytes)

        return render_template(
            "background_removal.html",
            input_b64=file_to_base64(input_path),
            output_b64=file_to_base64(output_path),
            title="Background Removal"
        )

    return render_template("background_removal.html")


# ----------------------------------------
# BACKGROUND REPLACE
# ----------------------------------------
@app.route("/background-replace", methods=["GET", "POST"])
def background_replace():

    mode = "color"
    selected_color = "#ffffff"

    if request.method == "POST":

        if "image" not in request.files:
            flash("Upload a main image.", "error")
            return redirect(request.url)

        main_file = request.files["image"]
        main_path = os.path.join(UPLOAD_DIR, main_file.filename)
        main_file.save(main_path)

        mode = request.form.get("mode")

        # COLOR REPLACE
        if mode == "color":
            selected_color = request.form.get("color", "#ffffff")

            output_path = os.path.join(OUTPUT_DIR, "replaced.png")
            replace_background_color(main_path, selected_color, save_to=output_path)

            return render_template(
                "background_replace.html",
                input_b64=file_to_base64(main_path),
                output_b64=file_to_base64(output_path),
                mode="color",
                selected_color=selected_color,
            )

        # IMAGE REPLACE
        if mode == "image":

            if "bg_image" not in request.files:
                flash("Upload background image.", "error")
                return redirect(request.url)

            bg_file = request.files["bg_image"]
            bg_path = os.path.join(UPLOAD_DIR, bg_file.filename)
            bg_file.save(bg_path)

            output_path = os.path.join(OUTPUT_DIR, "replaced.png")

            replace_background_image(main_path, bg_path, save_to=output_path)

            return render_template(
                "background_replace.html",
                input_b64=file_to_base64(main_path),
                output_b64=file_to_base64(output_path),
                mode="image"
            )

    return render_template("background_replace.html", mode="color", selected_color="#ffffff")


# ----------------------------------------
# BLUR BACKGROUND
# ----------------------------------------
@app.route("/blur", methods=["GET", "POST"])
def blur_background_page():

    if request.method == "POST":

        if "image" not in request.files:
            flash("Upload an image.", "error")
            return redirect(request.url)

        file = request.files["image"]
        img_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(img_path)

        blur_value = int(request.form.get("blur", 25))

        output_path = os.path.join(OUTPUT_DIR, "blurred.png")
        blur_background(img_path, blur_value, save_to=output_path)

        return render_template(
            "blur_background.html",
            input_b64=file_to_base64(img_path),
            output_b64=file_to_base64(output_path),
            blur_value=blur_value
        )

    return render_template("blur_background.html")

# ----------------------------------------
# BATCH PROCESSING
# ----------------------------------------
BATCH_ZIP_BUFFER = None


@app.route("/batch-processing", methods=["GET", "POST"])
def batch_processing():
    global BATCH_ZIP_BUFFER

    if request.method == "POST":
        if "images" not in request.files:
            return render_template("batch_processing.html", outputs=None)

        files = request.files.getlist("images")

        outputs = []
        zip_buffer = io.BytesIO()
        zip_file = zipfile.ZipFile(zip_buffer, "w")

        for f in files:
            try:
                # Load image
                img = Image.open(f.stream).convert("RGB")

                # Remove background
                result, _ = remove_background(img)

                # Convert to base64 for browser
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

                outputs.append({
                    "name": f.filename,
                    "b64": b64_str
                })

                # Add to zip
                zip_file.writestr(
                    f.filename.replace(".", "_processed."), 
                    buf.getvalue()
                )

            except Exception as e:
                print("Error processing:", f.filename, e)
                continue

        zip_file.close()
        zip_buffer.seek(0)
        BATCH_ZIP_BUFFER = zip_buffer  # store for download

        return render_template("batch_processing.html", outputs=outputs)

    return render_template("batch_processing.html", outputs=None)



@app.route("/download-zip")
def download_zip():
    global BATCH_ZIP_BUFFER
    if BATCH_ZIP_BUFFER is None:
        return redirect(url_for("batch_processing"))

    BATCH_ZIP_BUFFER.seek(0)
    return send_file(
        BATCH_ZIP_BUFFER,
        mimetype="application/zip",
        as_attachment=True,
        download_name="batch_processed.zip"
    )

# ----------------------------------------
# SAMPLE GALLERY
# ----------------------------------------
@app.route("/gallery")
def sample_gallery():
    files = os.listdir(SAMPLES_DIR)
    return render_template("sample_gallery.html", files=files)


# ----------------------------------------
# ABOUT PAGE
# ----------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# ----------------------------------------
# ENTRY POINT (Render uses Gunicorn so this rarely runs)
# ----------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

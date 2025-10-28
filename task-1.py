

# Install dependencies
!pip install pycocotools opencv-python matplotlib requests tqdm --quiet

import os
import zipfile
import requests
from tqdm.auto import tqdm
from pycocotools.coco import COCO
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# -------------------------------
# 1. Download COCO datasets
# -------------------------------

base_url = "http://images.cocodataset.org/zips/"
annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

downloads_dir = "downloads"
os.makedirs(downloads_dir, exist_ok=True)

datasets = {
    "train2017": base_url + "train2017.zip",
    "val2017": base_url + "val2017.zip"
}

# Download train and val images
for name, url in datasets.items():
    zip_path = os.path.join(downloads_dir, f"{name}.zip")
    extract_path = os.path.join(downloads_dir, name)
    if not os.path.exists(extract_path):
        print(f"Downloading {name} dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in tqdm(response.iter_content(8192)):
                f.write(chunk)
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(downloads_dir)
    else:
        print(f"{name} already exists.")

# Download annotations
annotations_zip_path = os.path.join(downloads_dir, "annotations_trainval2017.zip")
annotations_folder = os.path.join(downloads_dir, "annotations")
if not os.path.exists(annotations_folder):
    print("Downloading annotations...")
    response = requests.get(annotation_url, stream=True)
    with open(annotations_zip_path, "wb") as f:
        for chunk in tqdm(response.iter_content(8192)):
            f.write(chunk)
    with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
        zip_ref.extractall(downloads_dir)
else:
    print("Annotations already exist.")

# -------------------------------
# 2. Prepare COCO API
# -------------------------------
annotation_file = os.path.join(annotations_folder, "instances_train2017.json")
coco = COCO(annotation_file)

# -------------------------------
# 3. Create folders
# -------------------------------
train_folder = os.path.join(downloads_dir, "train2017")
output_folder = os.path.join(downloads_dir, "output_20000")
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 4. Process first 20,000 images
# -------------------------------
image_ids = coco.getImgIds()
print(f"Total available images in train2017: {len(image_ids)}")

limit = 20000  # Process first 20,000
output_paths = []

for img_id in tqdm(image_ids[:limit]):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    img_path = os.path.join(train_folder, file_name)

    if not os.path.exists(img_path):
        print(f"Image missing: {img_path}")
        continue

    # Load annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    if len(anns) == 0:
        continue

    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create mask
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    for ann in anns:
        rle = coco.annToMask(ann)
        mask = np.maximum(mask, rle * 255)

    # Apply mask: subject in color, background black
    mask_3ch = cv2.merge([mask, mask, mask])
    subject_color = cv2.bitwise_and(img_rgb, mask_3ch)

    # Save output
    output_file = os.path.join(output_folder, file_name)
    cv2.imwrite(output_file, cv2.cvtColor(subject_color, cv2.COLOR_RGB2BGR))
    output_paths.append(output_file)

print(f"✅ Saved {len(output_paths)} processed images (colored subject + black background) in '{output_folder}'")

# -------------------------------
# 5. Display few samples side by side
# -------------------------------
for output_file in output_paths[:50]:  # show 50 examples
    original_file = os.path.join(train_folder, os.path.basename(output_file))
    original = cv2.cvtColor(cv2.imread(original_file), cv2.COLOR_BGR2RGB)
    processed = cv2.cvtColor(cv2.imread(output_file), cv2.COLOR_BGR2RGB)
    combined = np.hstack((original, processed))
    plt.figure(figsize=(12, 6))
    plt.imshow(combined)
    plt.axis('off')
    plt.show()
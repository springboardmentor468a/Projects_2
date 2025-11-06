import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

IMG_PATH = r"C:\Users\Ayush Tiwari\Downloads\archive (1)\train_images"
MASK_PATH = r"C:\Users\Ayush Tiwari\Downloads\archive (1)\train_masks"

file = random.choice(os.listdir(IMG_PATH))
img_path = os.path.join(IMG_PATH, file)
mask_path = os.path.join(MASK_PATH, file.replace('.jpg', '.png'))

image = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

resize = transforms.Resize((128, 128))
image = resize(image)
mask = resize(mask)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.show()


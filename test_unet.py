import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from train_unet import UNet

model = UNet()
model.load_state_dict(torch.load("car_segmentation_model.pth", map_location=torch.device('cpu')))
model.eval()

image_path = "test_image.jpg"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
input_img = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_img)
    pred_mask = torch.sigmoid(output)
    pred_mask = (pred_mask > 0.5).float()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask[0][0], cmap='gray')
plt.title("Predicted Mask")
plt.show(block=True)

original = np.array(image)
mask = pred_mask[0][0].cpu().numpy()
mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

mask_colored = np.zeros_like(original)
mask_colored[:, :, 0] = mask * 255

overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)

plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title("Overlay (Mask on Image)")
plt.axis("off")
plt.show()

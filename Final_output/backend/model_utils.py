import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import segmentation

# ----------------- Config -----------------
FG_THRESHOLD = 0.6
MORPH_KERNEL = 7
MORPH_ITER = 3
KEEP_LARGEST = True

# ----------------- Model loader -----------------
@torch.no_grad()
def load_model(device="cpu"):
    model = segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
    model.eval()
    model.to(device)
    return model

# ----------------- Preprocessing -----------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_to_tensor(img_pil: Image.Image, device="cpu"):
    return preprocess(img_pil).unsqueeze(0).to(device)

# ----------------- Mask Helpers -----------------
def keep_largest_component(mask_np):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, 8)
    if num_labels <= 1:
        return mask_np
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype("uint8") * 255

def refine_mask(prob_map):
    # convert to binary mask
    bin_mask = (prob_map >= FG_THRESHOLD).astype("uint8") * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_KERNEL, MORPH_KERNEL))

    # close holes + open noise
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)

    if KEEP_LARGEST:
        cleaned = keep_largest_component(cleaned)

    return cleaned

# ----------------- Core Segmentation -----------------
def segment_object_only(img_pil: Image.Image, model, device="cpu", bg_color=(0, 0, 0)):
    """Return image with subject unchanged and background pure black."""
    
    # Step 1: model forward
    tensor = image_to_tensor(img_pil, device)
    with torch.no_grad():
        out = model(tensor)

    # Step 2: get probability map
    logits = out['out'][0]
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

    # foreground = any class > background
    fg_prob = np.max(probs[1:], axis=0)

    # resize probability to original size
    img_w, img_h = img_pil.size
    fg_prob_resized = cv2.resize(fg_prob, (img_w, img_h), cv2.INTER_NEAREST)

    # Step 3: refine mask
    bin_mask = refine_mask(fg_prob_resized)

    # Step 4: apply mask — background becomes black
    src = np.array(img_pil)

    # mask 0 or 1
    mask = (bin_mask / 255).astype("float32")
    mask = mask[:, :, None]

    # pure black background
    background = np.zeros_like(src)  # (H, W, 3)

    # combine: subject stays, background becomes black
    out_rgb = src * mask + background * (1 - mask)

    return Image.fromarray(out_rgb.astype("uint8"))

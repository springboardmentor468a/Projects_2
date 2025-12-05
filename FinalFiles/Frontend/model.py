import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image

# -------------------------
#   Your Original Model
# -------------------------
class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,   # MUST BE NONE when loading checkpoint
            in_channels=3,
            classes=1
        )

        self.extra = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.extra(x)
        return x


# -------------------------
# Load Your Trained Model
# -------------------------
def load_model(path):
    model = CustomUNet()
    state_dict = torch.load(path, map_location="cpu")

    # REMOVE 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


# -------------------------
# Preprocessing
# -------------------------
def preprocess(image):
    image = image.resize((256, 256))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    tensor = torch.tensor(arr).unsqueeze(0)
    return tensor


# -------------------------
# Prediction + Resize Mask
# -------------------------
def get_segmented_output(model, image):

    orig_w, orig_h = image.size

    inp = preprocess(image)
    with torch.no_grad():
        mask = model(inp)[0][0].numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize((orig_w, orig_h))

    # Overlay
    overlay = image.copy()
    overlay_np = np.array(overlay)
    mask_np = np.array(mask_img)

    overlay_np[mask_np == 0] = 0
    overlay = Image.fromarray(overlay_np)

    return mask_img, overlay

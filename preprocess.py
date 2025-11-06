import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CarSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_stem = os.path.splitext(img_name)[0]

        possible_mask = None
        for ext in ['.png', '.jpg', '.jpeg']:
            name_options = [
                img_stem + ext,
                img_stem + '_mask' + ext,
                img_stem + '-mask' + ext,
                img_stem + ' mask' + ext
            ]
            for name in name_options:
                mask_path = os.path.join(self.mask_dir, name)
                if os.path.exists(mask_path):
                    possible_mask = mask_path
                    break
            if possible_mask:
                break

        if not possible_mask:
            print("DEBUG:", img_name)
            print("Looking for mask in:", self.mask_dir)
            raise FileNotFoundError(f"❌ Mask not found for image: {img_name}")

        mask_path = possible_mask

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Resize((128, 128))(image)

        mask = transforms.Resize((128, 128))(mask)
        mask = transforms.ToTensor()(mask)

        return image, mask

# update for consistency

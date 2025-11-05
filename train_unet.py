import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocess import CarSegmentationDataset
import numpy as np
import matplotlib.pyplot as plt
from model import UNet

print("Training script started... loading dataset")

IMG_PATH = r"C:\Users\Ayush Tiwari\OneDrive\Desktop\CarSegmentation\dataset\train_images"
MASK_PATH = r"C:\Users\Ayush Tiwari\OneDrive\Desktop\CarSegmentation\dataset\train_masks"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("Loading dataset... please wait")
dataset = CarSegmentationDataset(IMG_PATH, MASK_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print("Dataset loaded successfully... now training will start")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, 2, s

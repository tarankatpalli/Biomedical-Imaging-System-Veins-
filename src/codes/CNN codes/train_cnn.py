import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import VeinCNN

FRAMES_DIR = "/home/taran/vein-t/data/frames"
MASKS_DIR = "/home/taran/vein-t/data/masks"
IMG_SIZE = 256
EPOCHS = 20
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class VeinDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to load {img_path} or {mask_path}")

        # Resize BOTH to same size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img = img / 255.0
        mask = mask / 255.0

        # Add channel dimension
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask

dataset = VeinDataset(FRAMES_DIR, MASKS_DIR)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0   # REQUIRED for Raspberry Pi
)

model = VeinCNN().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#training loop
try:
    for epoch in range(EPOCHS):
        print(f"\nStarting epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0

        for i, (x, y) in enumerate(loader):
            print(f"  Batch {i}")

            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")

os.makedirs("/home/taran/vein-t/output", exist_ok=True)
torch.save(model.state_dict(), "/home/taran/vein-t/output/cnn_veins.pth")
print("Model saved to output/cnn_veins.pth")

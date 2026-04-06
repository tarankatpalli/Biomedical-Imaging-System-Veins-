import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = "/home/taran/vein-t/hand.jpg" #Update this
OUT_DIR = Path("/home/taran/vein-t/stage1_output") #Update this
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"  # RPI-safe

print("Loading image...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Failed to load image")

h, w, _ = img.shape
print(f"Loaded image: {w}x{h}")

# Convert BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Running MiDaS depth estimation...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(DEVICE)
midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS", "transforms"
)
transform = midas_transforms.small_transform

# IMPORTANT: transform already adds batch dim
input_batch = transform(img_rgb).to(DEVICE)

with torch.no_grad():
    depth = midas(input_batch)

depth = depth.squeeze().cpu().numpy()

# Normalize depth for visualization
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = depth_norm.astype(np.uint8)

cv2.imwrite(str(OUT_DIR / "depth.png"), depth_norm)
print("Depth map saved")

print("Running classical vein detection...")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
vein_enhanced = clahe.apply(gray)

# Edge detection (Canny)
edges = cv2.Canny(vein_enhanced, 50, 150)

# Save vein images
cv2.imwrite(str(OUT_DIR / "gray.png"), gray)
cv2.imwrite(str(OUT_DIR / "clahe.png"), vein_enhanced)
cv2.imwrite(str(OUT_DIR / "edges.png"), edges)

contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
for c in contours:
    c = c.squeeze()
    if len(c.shape) == 2:
        plt.plot(c[:, 0], c[:, 1], linewidth=0.5)

plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "vein_contours.png", dpi=300)
plt.close()

print("Generating hand mask...")

# Threshold CLAHE image
_, thresh = cv2.threshold(vein_enhanced, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological cleanup
kernel = np.ones((7, 7), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

# Find connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean)

# Largest component (excluding background)
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
mask = np.zeros_like(clean)
mask[labels == largest] = 255

cv2.imwrite(str(OUT_DIR / "mask.png"), mask)

print("Hand mask saved")

print("Vein processing complete")

print("\nStage 1 COMPLETE")
print(f"Outputs saved to: {OUT_DIR}")

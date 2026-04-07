import os
import cv2
import numpy as np
from skimage.filters import frangi

FRAMES_DIR = "/home/taran/vein-t/data/frames"
MASKS_DIR  = "/home/taran/vein-t/data/masks"

IMG_SIZE = 256

os.makedirs(MASKS_DIR, exist_ok=True)

files = sorted(os.listdir(FRAMES_DIR))
print(f"[INFO] Found {len(files)} files in frames folder")

count = 0

for name in files:
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(FRAMES_DIR, name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[WARN] Could not read {name}")
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    #Frangi Vessel Enhancement
    vessels = frangi(img, scale_range=(1, 3), scale_step=1)
    vessels = (vessels / vessels.max() * 255).astype(np.uint8)

    save_path = os.path.join(MASKS_DIR, name)
    cv2.imwrite(save_path, vessels)

    count += 1

print(f"[DONE] Generated {count} masks in {MASKS_DIR}")

from train_cnn import VeinDataset

FRAMES_DIR = "/home/taran/vein-t/data/frames"
MASKS_DIR = "/home/taran/vein-t/data/masks"

ds = VeinDataset(FRAMES_DIR, MASKS_DIR)

print("Dataset size:", len(ds))

x, y = ds[0]
print("Image shape:", x.shape)
print("Mask shape:", y.shape)

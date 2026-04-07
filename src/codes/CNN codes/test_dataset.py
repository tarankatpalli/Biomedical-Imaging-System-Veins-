from train_cnn import VeinDataset

FRAMES_DIR = "/home/taran/vein-t/data/frames" #Change this (depending on organization)
MASKS_DIR = "/home/taran/vein-t/data/masks" #Change this (depending on organization)

ds = VeinDataset(FRAMES_DIR, MASKS_DIR)

print("Dataset size:", len(ds))

x, y = ds[0]
print("Image shape:", x.shape)
print("Mask shape:", y.shape)

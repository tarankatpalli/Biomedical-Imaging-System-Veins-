import cv2
import numpy as np
import open3d as o3d

depth_img = cv2.imread("/home/taran/vein-t/stage1_output/depth.png", cv2.IMREAD_GRAYSCALE) # Change this
mask_img = cv2.imread("/home/taran/vein-t/stage1_output/vein_contours.png", cv2.IMREAD_GRAYSCALE) # Change this

if depth_img is None or mask_img is None:
    raise RuntimeError("Missing depth.png or mask.png")

h, w = depth_img.shape

# Normalize depth to meters (relative scale)
depth = depth_img.astype(np.float32) / 255.0
depth = depth * 0.5  # 0–50cm working volume

mask = mask_img > 0

fx = fy = 550.0
cx = w / 2
cy = h / 2

points = []

print("Generating 3D points...")
for y in range(h):
    for x in range(w):
        if not mask[y, x]:
            continue
        z = depth[y, x]
        if z <= 0:
            continue
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        points.append([X, -Y, z])  # flip Y for 3D coords

points = np.array(points, dtype=np.float64)

print(f"Raw point count: {len(points)}")

MAX_POINTS = 20000
if len(points) > MAX_POINTS:
    idx = np.random.choice(len(points), MAX_POINTS, replace=False)
    points = points[idx]
    print(f"Subsampled to {len(points)} points for stability")

if len(points) == 0:
    raise RuntimeError("No valid 3D points generated")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

#Voxel Downsampling
pcd = pcd.voxel_down_sample(voxel_size=0.005)
print(f"After voxel downsampling: {len(pcd.points)}")

#Statisical Outlier Removal
pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)
print(f"After outlier removal: {len(pcd.points)}")

o3d.io.write_point_cloud("hand_cloud.ply", pcd)
print("Stage 2 COMPLETE → hand_cloud.ply")

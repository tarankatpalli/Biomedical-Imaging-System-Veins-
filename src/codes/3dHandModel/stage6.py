import open3d as o3d
import numpy as np
import cv2
from scipy.spatial import cKDTree
import trimesh
from skimage.morphology import skeletonize

MESH_FILE = "/home/taran/vein-t/final_hand.ply"
VEIN_IMAGE = "/home/taran/vein-t/stage1_output/vein_contours.png"  # adjust if needed

OUTPUT_PLY = "final_hand_with_vein_lines.ply"
OUTPUT_STL = "final_hand_with_vein_lines.stl"

VEIN_COLOR = np.array([1.0, 0.0, 0.0])      # Red lines
SKIN_COLOR = np.array([0.85, 0.75, 0.65])   # Neutral skin

LINE_DISTANCE_PIXELS = 2   # how close a vertex must be to vein centerline

print("Loading mesh...")
mesh = o3d.io.read_triangle_mesh(MESH_FILE)
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)

print("Loading vein image...")
img = cv2.imread(VEIN_IMAGE, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load {VEIN_IMAGE}")

#Binary Mask Vein
_, vein_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
vein_bin = vein_bin.astype(bool)

print("Skeletonizing vein map...")
vein_skel = skeletonize(vein_bin)

# Get skeleton pixel coordinates
skel_pixels = np.column_stack(np.where(vein_skel))
if len(skel_pixels) == 0:
    raise RuntimeError("No vein skeleton pixels detected")

print(f"Skeleton pixels: {len(skel_pixels)}")

# KD-tree for distance queries
tree = cKDTree(skel_pixels)

h, w = img.shape

# Normalize mesh XY → image coords
x = vertices[:, 0]
y = vertices[:, 1]

x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

u = (x_norm * (w - 1)).astype(int)
v = ((1 - y_norm) * (h - 1)).astype(int)

# Assign colors
colors = np.tile(SKIN_COLOR, (len(vertices), 1))
line_count = 0

print("Projecting vein centerlines onto mesh...")
for i in range(len(vertices)):
    dist, _ = tree.query([v[i], u[i]])
    if dist <= LINE_DISTANCE_PIXELS:
        colors[i] = VEIN_COLOR
        line_count += 1

print(f"Marked {line_count} vertices as vein lines")

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

print("Saving PLY with vein lines...")
o3d.io.write_triangle_mesh(OUTPUT_PLY, mesh)

print("Saving STL (geometry only)...")
tm = trimesh.Trimesh(
    vertices=np.asarray(mesh.vertices),
    faces=np.asarray(mesh.triangles),
    process=True
)
tm.export(OUTPUT_STL)

print("\nFINAL STAGE COMPLETE")
print(f"PLY (vein centerlines): {OUTPUT_PLY}")
print(f"STL (geometry): {OUTPUT_STL}")

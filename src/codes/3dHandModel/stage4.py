import torch
import smplx
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import trimesh

MANO_PATH = "/home/taran/vein-t/mano/models"   # folder containing MANO_*.pkl          Download yourself and Change this
HAND_SIDE = "left"              # "right" or "left"
POINT_CLOUD = "/home/taran/vein-t/hand_cloud_clean.ply" #Change this

DEVICE = torch.device("cpu")     # RPi-safe
NUM_ITERS = 200
LR = 0.01

print("Loading cleaned point cloud...")
pcd = o3d.io.read_point_cloud(POINT_CLOUD)
pcd = pcd.voxel_down_sample(0.007)
points = np.asarray(pcd.points)

print(f"Using {len(points)} points")

points_torch = torch.tensor(points, dtype=torch.float32, device=DEVICE)

print("Loading MANO model...")
mano = smplx.create(
    MANO_PATH,
    model_type="mano",
    is_rhand=(HAND_SIDE == "right"),
    use_pca=True,
    num_pca_comps=12,
    flat_hand_mean=True
).to(DEVICE)

hand_pose = torch.zeros(
    [1, mano.num_pca_comps], device=DEVICE, requires_grad=True
)
betas = torch.zeros([1, 10], device=DEVICE, requires_grad=True)
global_orient = torch.zeros([1, 3], device=DEVICE, requires_grad=True)
transl = torch.zeros([1, 3], device=DEVICE, requires_grad=True)

optimizer = torch.optim.Adam(
    [hand_pose, betas, global_orient, transl], lr=LR
)

print("Fitting MANO to point cloud...")

best_loss = float("inf")
patience = 15       # number of iterations to wait for improvement
counter = 0

for i in range(NUM_ITERS):
    optimizer.zero_grad()

    output = mano(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        transl=transl
    )

    verts = output.vertices.squeeze(0)

    diff = verts.unsqueeze(1) - points_torch.unsqueeze(0)  
    dists = torch.norm(diff, dim=2)                         
    min_dists, _ = torch.min(dists, dim=1)                  
    loss = torch.mean(min_dists)

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Iter {i:03d} | Loss: {loss.item():.6f}")

    if loss.item() < best_loss - 1e-5:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at iteration {i}, loss stabilized at {best_loss:.6f}")
            break

print("Exporting MANO mesh...")

verts = output.vertices.squeeze(0).detach().cpu().numpy()
faces = mano.faces

mesh = trimesh.Trimesh(
    vertices=verts,
    faces=faces,
    process=False
)

mesh.export("mano_fitted.ply")

print("\n Stage 3 COMPLETE")
print("Saved → mano_fitted.ply")

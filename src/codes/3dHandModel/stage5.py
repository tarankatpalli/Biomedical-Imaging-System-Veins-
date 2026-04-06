import open3d as o3d
import trimesh
import numpy as np

INPUT_MESH = "/home/taran/vein-t/mano_fitted.ply" # Change this
OUTPUT_PLY = "final_hand.ply"
OUTPUT_STL = "final_hand.stl"

# Smoothing parameters
LAPLACIAN_ITER = 30
EDGE_PRESERVE_ITER = 5
EDGE_PRESERVE_ALPHA = 0.5  # 0.0 = no smoothing, 1.0 = full smoothing

print("Loading fitted MANO mesh...")
mesh_o3d = o3d.io.read_triangle_mesh(INPUT_MESH)

# Ensure normals exist
mesh_o3d.compute_vertex_normals()

print("Applying Laplacian smoothing...")
mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=LAPLACIAN_ITER)

print("Applying edge-preserving smoothing...")
mesh_o3d = mesh_o3d.filter_smooth_taubin(number_of_iterations=EDGE_PRESERVE_ITER, lambda_filter=EDGE_PRESERVE_ALPHA)

print("Computing vertex normals...")
mesh_o3d.compute_vertex_normals()

# Save final PLY
o3d.io.write_triangle_mesh(OUTPUT_PLY, mesh_o3d)
print(f"Saved smoothed mesh → {OUTPUT_PLY}")

# Convert to STL using trimesh
mesh_trimesh = trimesh.Trimesh(
    vertices=np.asarray(mesh_o3d.vertices),
    faces=np.asarray(mesh_o3d.triangles),
    process=True
)
mesh_trimesh.export(OUTPUT_STL)
print(f"Saved STL mesh → {OUTPUT_STL}")

print("\nStage 4 COMPLETE — Mesh is smoothed and ready!")

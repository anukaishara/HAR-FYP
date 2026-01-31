import torch
import numpy as np
import smplx
import trimesh
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FOLDER = 'model_data'
PROFILE_FILE = 'user_profile.npy'
GENDER = 'neutral'

# ==========================================
# 1. LOAD PROFILE
# ==========================================
if not os.path.exists(PROFILE_FILE):
    print("❌ Profile not found.")
    exit()

user_data = np.load(PROFILE_FILE, allow_pickle=True).item()
betas = torch.from_numpy(user_data['betas']).float().unsqueeze(0)
scale = user_data.get('scale', 1.0)
print(f"✅ Loaded Profile. Scale factor: {scale:.4f}")

# ==========================================
# 2. LOAD MODEL
# ==========================================
smpl = smplx.SMPL(model_path=MODEL_FOLDER, gender=GENDER)

# ==========================================
# 3. GENERATE MESH
# ==========================================
print("generating mesh...")
output = smpl(betas=betas, return_verts=True)
vertices = output.vertices.detach().cpu().numpy()[0]
faces = smpl.faces

# ==========================================
# 4. APPLY ENGINEERING VISUALS
# ==========================================
mesh = trimesh.Trimesh(vertices, faces)

# COLOR: Standard "Silver/Clay" Grey
# RGB: [192, 192, 192] + Alpha: [255]
mesh.visual.vertex_colors = [192, 192, 192, 255]

# ==========================================
# 5. SHOW IN BLACK VOID
# ==========================================
print("🎥 Opening Silver Viewer...")
scene = trimesh.Scene(mesh)

# OPTIONAL: Add a faint grid floor for perspective
# grid = trimesh.creation.grid(size=4, sections=10)
# grid.visual.vertex_colors = [50, 50, 50, 100] # Dark grey grid
# scene.add_geometry(grid)

# Show with BLACK background [0, 0, 0, 255]
scene.show(background=[0, 0, 0, 255])
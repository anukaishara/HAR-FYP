import torch
import numpy as np
import smplx
import trimesh
import os
import time
import glob
import sys
from scipy.spatial.transform import Rotation as R

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FOLDER = 'model_data'
PROFILE_FILE = 'user_profile.npy'
OUTPUT_DIR = 'outputs'
GENDER = 'neutral'
INTERPOLATION_FACTOR = 2  # 2x smoother (30fps -> 60fps)

# ==========================================
# 1. LOAD AVATAR & DATA
# ==========================================
if not os.path.exists(PROFILE_FILE):
    print("❌ Profile not found.")
    exit()

user_data = np.load(PROFILE_FILE, allow_pickle=True).item()
user_betas = torch.from_numpy(user_data['betas']).float().unsqueeze(0)
smpl = smplx.SMPL(model_path=MODEL_FOLDER, gender=GENDER)

list_of_files = glob.glob(os.path.join(OUTPUT_DIR, '*.npz'))
if not list_of_files:
    print(f"❌ No .npz files found in {OUTPUT_DIR}")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"📂 Scanning: {latest_file}")
data = np.load(latest_file, allow_pickle=True)

# Universal Hunter
candidates = []
def recursive_scan(obj):
    if isinstance(obj, dict):
        for k in sorted(obj.keys()): recursive_scan(obj[k])
    elif isinstance(obj, (list, tuple)):
        for item in obj: recursive_scan(item)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        try:
            arr = torch.tensor(obj).float()
            if arr.numel() == 72: candidates.append(arr.view(1, 72))
            elif arr.ndim == 2 and arr.shape[1] == 72: candidates.append(arr)
        except: pass

if 'results' in data:
    recursive_scan(data['results'].item() if hasattr(data['results'], 'item') else data['results'])
else:
    recursive_scan(data)

if not candidates:
    print("❌ CRITICAL: No pose data found.")
    exit()

big_chunks = [c for c in candidates if c.shape[0] > 1]
if big_chunks:
    raw_pose_seq = max(big_chunks, key=lambda x: x.shape[0])
else:
    raw_pose_seq = torch.stack([c[0] for c in candidates])

print(f"✅ Found {raw_pose_seq.shape[0]} raw frames.")

# ==========================================
# 2. INTERPOLATE (Make it Smooth)
# ==========================================
print(f"🔹 Interpolating to {INTERPOLATION_FACTOR}x framerate...")

# Convert to Numpy for interpolation
raw_poses = raw_pose_seq.numpy()
orig_frames = np.arange(len(raw_poses))
new_frames = np.linspace(0, len(raw_poses)-1, len(raw_poses) * INTERPOLATION_FACTOR)

# Linear Interpolation of Poses
pose_seq_interp = np.zeros((len(new_frames), 72))
for i in range(72):
    pose_seq_interp[:, i] = np.interp(new_frames, orig_frames, raw_poses[:, i])

pose_seq = torch.from_numpy(pose_seq_interp).float()
num_frames = pose_seq.shape[0]
print(f"✅ SMOOTHED: {num_frames} frames ready.")

# ==========================================
# 3. PRE-BAKE (Calculate Physics)
# ==========================================
print("\n🔥 BAKING ANIMATION... (Calculating Physics)")
prebaked_verts = []
R_fix = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])

start_bake = time.time()
for i in range(num_frames):
    current_pose = pose_seq[i].unsqueeze(0)
    with torch.no_grad():
        output = smpl(betas=user_betas, body_pose=current_pose[:, 3:], global_orient=current_pose[:, :3], return_verts=True)
    v = output.vertices.detach().cpu().numpy()[0]
    v = v @ R_fix.T   # Flip Up
    v[:, 1] += 1.05   # Lift to Floor
    prebaked_verts.append(v)
    
    if i % 50 == 0:
        sys.stdout.write(f"\r   ⏳ Baking: {i/num_frames*100:.0f}%")

print(f"\n✅ Baking Complete! ({time.time() - start_bake:.2f}s)")

# ==========================================
# 4. HIGH PERFORMANCE PLAYBACK
# ==========================================
print("\n🎥 Launching Smooth 60FPS Player...")

# Setup Scene
mesh = trimesh.Trimesh(prebaked_verts[0], smpl.faces)
mesh.visual.vertex_colors = [192, 192, 192, 255] # Silver
scene = trimesh.Scene(mesh)

# Floor
lines = []
for i in range(-5, 6):
    lines.append([[i, 0, -5], [i, 0, 5]])
    lines.append([[-5, 0, i], [5, 0, i]])
for line in lines:
    path = trimesh.load_path(np.array(line))
    path.colors = [np.array([50, 50, 50, 255]) for _ in range(len(path.entities))]
    scene.add_geometry(path)

frame_idx = 0

def update_scene(scene):
    global frame_idx
    
    idx = frame_idx % num_frames
    
    # Instant Geometry Swap
    mesh_name = list(scene.geometry.keys())[0]
    scene.geometry[mesh_name].vertices = prebaked_verts[idx]
    
    frame_idx += 1
    # NO SLEEP -> Max smoothness

# Launch
scene.show(callback=update_scene, background=[0,0,0,255])
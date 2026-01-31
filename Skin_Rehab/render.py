import torch
import numpy as np
import smplx
import trimesh
import os
import time
import glob
import sys

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FOLDER = 'C:\Users\Anuka\model_data'
PROFILE_FILE = 'user_profile.npy'
OUTPUT_DIR = 'outputs'
GENDER = 'neutral'
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS

# ==========================================
# 1. LOAD DATA
# ==========================================
if not os.path.exists(PROFILE_FILE):
    print("❌ Profile not found.")
    exit()

user_data = np.load(PROFILE_FILE, allow_pickle=True).item()
user_betas = torch.from_numpy(user_data['betas']).float().unsqueeze(0)
smpl = smplx.SMPL(model_path=MODEL_FOLDER, gender=GENDER)

list_of_files = glob.glob(os.path.join(OUTPUT_DIR, '*.npz'))
if not list_of_files:
    print(f"❌ No .npz files found.")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
data = np.load(latest_file, allow_pickle=True)

# --- UNIVERSAL HUNTER ---
candidates = []
def recursive_scan(obj):
    if isinstance(obj, dict):
        # Sort keys to ensure frames stay in order (0, 1, 2...)
        for k in sorted(obj.keys()): recursive_scan(obj[k])
    elif isinstance(obj, (list, tuple)):
        for item in obj: recursive_scan(item)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        try:
            arr = torch.tensor(obj).float()
            # Catch single frames (1, 72) or (72,)
            if arr.numel() == 72: 
                candidates.append(arr.view(1, 72))
            # Catch videos (N, 72)
            elif arr.ndim == 2 and arr.shape[1] == 72: 
                candidates.append(arr)
        except: pass

if 'results' in data:
    recursive_scan(data['results'].item() if hasattr(data['results'], 'item') else data['results'])
else:
    recursive_scan(data)

if not candidates:
    print("❌ CRITICAL: No pose data found.")
    exit()

# --- THE FIX: STITCHING LOGIC ---
# Check if we have any "Big Chunks" (Video format)
big_chunks = [c for c in candidates if c.shape[0] > 1]

if big_chunks:
    # If we found a video file, take the biggest one
    raw_pose_seq = max(big_chunks, key=lambda x: x.shape[0])
    print("🔹 Mode: Video Block Detected")
else:
    # If we only found single frames, STACK them together
    raw_pose_seq = torch.cat(candidates, dim=0)
    print("🔹 Mode: Frame Stitching Detected")

print(f"✅ Loaded {raw_pose_seq.shape[0]} raw frames.")

# ==========================================
# 2. INTERPOLATE TO 60 FPS
# ==========================================
current_fps = 30
interp_factor = TARGET_FPS / current_fps 

print(f"🔹 Interpolating to {TARGET_FPS} FPS...")
raw_poses = raw_pose_seq.numpy()
old_indices = np.arange(len(raw_poses))
new_indices = np.linspace(0, len(raw_poses)-1, int(len(raw_poses) * interp_factor))

pose_seq_interp = np.zeros((len(new_indices), 72))
for i in range(72):
    pose_seq_interp[:, i] = np.interp(new_indices, old_indices, raw_poses[:, i])

pose_seq = torch.from_numpy(pose_seq_interp).float()
num_frames = pose_seq.shape[0]

# ==========================================
# 3. PRE-BAKE PHYSICS
# ==========================================
print("🔥 BAKING PHYSICS (Pre-calculating)...")
prebaked_verts = []
R_fix = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])

start_bake = time.time()
for i in range(num_frames):
    with torch.no_grad():
        output = smpl(betas=user_betas, body_pose=pose_seq[i:i+1, 3:], global_orient=pose_seq[i:i+1, :3], return_verts=True)
    v = output.vertices.detach().cpu().numpy()[0]
    v = v @ R_fix.T 
    v[:, 1] += 1.05 
    prebaked_verts.append(v)
    
    if i % 50 == 0:
        sys.stdout.write(f"\r   ⏳ Progress: {i/num_frames*100:.0f}%")

print(f"\n✅ Ready! ({time.time() - start_bake:.2f}s)")

# ==========================================
# 4. HIGH-PRECISION GAME LOOP
# ==========================================
print(f"\n🎥 Launching Precision Player ({TARGET_FPS} FPS)...")

mesh = trimesh.Trimesh(prebaked_verts[0], smpl.faces)
mesh.visual.vertex_colors = [192, 192, 192, 255]
scene = trimesh.Scene(mesh)

lines = []
for i in range(-5, 6):
    lines.append([[i, 0, -5], [i, 0, 5]])
    lines.append([[-5, 0, i], [5, 0, i]])
for line in lines:
    path = trimesh.load_path(np.array(line))
    path.colors = [np.array([50, 50, 50, 255]) for _ in range(len(path.entities))]
    scene.add_geometry(path)

frame_idx = 0
start_time = time.time()

def update_scene(scene):
    global frame_idx
    target_time = start_time + (frame_idx * FRAME_TIME)
    if time.time() < target_time:
        return 
        
    idx = frame_idx % num_frames
    mesh_name = list(scene.geometry.keys())[0]
    scene.geometry[mesh_name].vertices = prebaked_verts[idx]
    frame_idx += 1

scene.show(callback=update_scene, background=[0,0,0,255])
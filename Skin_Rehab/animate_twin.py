import torch
import numpy as np
import smplx
import trimesh
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FOLDER = 'model_data'
PROFILE_FILE = 'user_profile.npy'
GENDER = 'neutral'

# ==========================================
# 1. LOAD PROFILE & MODEL
# ==========================================
if not os.path.exists(PROFILE_FILE):
    print("❌ Profile not found.")
    exit()

user_data = np.load(PROFILE_FILE, allow_pickle=True).item()
betas = torch.from_numpy(user_data['betas']).float().unsqueeze(0)

# Load SMPL
smpl = smplx.SMPL(model_path=MODEL_FOLDER, gender=GENDER)
print("✅ Digital Twin Loaded.")

# ==========================================
# 2. PRE-CALCULATE ANIMATION (Thetas)
# ==========================================
# We generate 60 frames of "waving" motion
frames = 60
pose_sequence = []
for i in range(frames):
    pose = torch.zeros((1, 72))
    # Waving motion
    angle = np.sin(i / 5.0) * 0.6 
    pose[0, 48+2] = angle   # Left Arm
    pose[0, 51+2] = -angle  # Right Arm
    pose_sequence.append(pose)

# ==========================================
# 3. SETUP SCENE
# ==========================================
# Create the initial mesh (Frame 0)
output = smpl(betas=betas, body_pose=pose_sequence[0][:, 3:], global_orient=pose_sequence[0][:, :3], return_verts=True)
initial_verts = output.vertices.detach().cpu().numpy()[0]
mesh = trimesh.Trimesh(initial_verts, smpl.faces)

# --- VISUAL STYLE: ENGINEERING SILVER ---
# [192, 192, 192] is standard silver/grey
mesh.visual.vertex_colors = [192, 192, 192, 255] 

scene = trimesh.Scene(mesh)
frame_idx = 0

# ==========================================
# 4. ANIMATION CALLBACK
# ==========================================
def update_scene(scene):
    global frame_idx
    
    # Get the pose for the current frame (looping)
    current_pose = pose_sequence[frame_idx % frames]
    
    # Run SMPL Math to get new shape
    output = smpl(betas=betas, 
                  body_pose=current_pose[:, 3:], 
                  global_orient=current_pose[:, :3], 
                  return_verts=True)
    
    # Update the mesh vertices in-place
    new_verts = output.vertices.detach().cpu().numpy()[0]
    
    mesh_name = list(scene.geometry.keys())[0]
    scene.geometry[mesh_name].vertices = new_verts
    
    frame_idx += 1
    time.sleep(0.03) # Cap at ~30 FPS for smoothness

# ==========================================
# 5. START VIEWER (BLACK BACKGROUND)
# ==========================================
print("🎥 Animation running... (Close window to stop)")
# background=[0, 0, 0, 255] sets it to solid black
scene.show(callback=update_scene, background=[0, 0, 0, 255])
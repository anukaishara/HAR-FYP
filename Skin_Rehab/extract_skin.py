import torch
import numpy as np
import smplx
import cv2
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_IMAGE_PATH = "inputs/internet_test.jpg" 
OUTPUT_DIR = "outputs/"
PROFILE_FILE = "user_profile.npy"
MODEL_FOLDER = "C:\Users\Anuka\model_data
GENDER = "neutral"

# ==========================================
# 1. LOAD DATA & IMAGE
# ==========================================
if not os.path.exists(PROFILE_FILE):
    print("❌ ERROR: 'user_profile.npy' not found. Run Step 1 first.")
    exit()

print(f"📸 Loading Image: {INPUT_IMAGE_PATH}")
image = cv2.imread(INPUT_IMAGE_PATH)
if image is None:
    print(f"❌ ERROR: Could not load image. Check path: {INPUT_IMAGE_PATH}")
    exit()

img_h, img_w = image.shape[:2]

user_data = np.load(PROFILE_FILE, allow_pickle=True).item()
betas = torch.from_numpy(user_data['betas']).float().unsqueeze(0)

# ==========================================
# 2. GET CAMERA INFO (ROBUST LOADING)
# ==========================================
result_files = glob.glob(os.path.join(OUTPUT_DIR, "*.npz"))
if not result_files:
    print("❌ ERROR: No ROMP output found.")
    exit()

latest_file = max(result_files, key=os.path.getctime)
print(f"📂 Reading data from: {latest_file}")
data = np.load(latest_file, allow_pickle=True)
results = data['results'].item() if hasattr(data['results'], 'item') else data['results'][0]

# --- FIX START: Handle different key names ---
# 1. Get Camera
if 'cam' in results:
    cam_params = results['cam'][0]
elif 'cam_trans' in results:
    cam_params = results['cam_trans'][0] # Some versions use this
else:
    print("❌ Critical Error: Could not find Camera parameters in file.")
    print(f"Available keys: {results.keys()}")
    exit()

# 2. Get Pose (The part that failed before)
pose = None
possible_keys = ['pose', 'smpl_thetas', 'thetas', 'params']

for key in possible_keys:
    if key in results:
        print(f"✅ Found pose data under key: '{key}'")
        pose = torch.tensor(results[key][0]).unsqueeze(0).float()
        break

if pose is None:
    print("❌ Critical Error: Could not find Pose data.")
    print(f"Available keys in file: {list(results.keys())}")
    exit()
# --- FIX END ---

global_orient = pose[:, :3]
body_pose = pose[:, 3:]

# ==========================================
# 3. RECONSTRUCT 3D MESH
# ==========================================
smpl = smplx.SMPL(model_path=MODEL_FOLDER, gender=GENDER)
output = smpl(betas=betas, global_orient=global_orient, body_pose=body_pose, return_verts=True)
vertices = output.vertices.detach().cpu().numpy()[0]

# ==========================================
# 4. PROJECT 3D TO 2D
# ==========================================
print("🎨 Projecting pixels...")
s, tx, ty = cam_params

# ROMP Projection Math
projected_verts = vertices.copy()
projected_verts[:, 0] = s * (vertices[:, 0] + tx)
projected_verts[:, 1] = s * (vertices[:, 1] + ty)

u = (projected_verts[:, 0] + 1) * 0.5 * img_w
v = (projected_verts[:, 1] + 1) * 0.5 * img_h

u = np.clip(u, 0, img_w - 1).astype(int)
v = np.clip(v, 0, img_h - 1).astype(int)

# ==========================================
# 5. SAMPLE COLORS & SAVE
# ==========================================
vertex_colors = image[v, u, ::-1] # BGR -> RGB
alpha = np.ones((vertex_colors.shape[0], 1)) * 255
vertex_colors = np.hstack([vertex_colors, alpha]).astype(np.uint8)

user_data['skin_colors'] = vertex_colors
np.save(PROFILE_FILE, user_data)

print("\n" + "="*40)
print("✅ SKIN EXTRACTED SUCCESSFULLY")
print(f"Data updated in: {PROFILE_FILE}")
print("="*40)
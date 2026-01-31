import os
import numpy as np
import glob
import subprocess

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
INPUT_FILE = "inputs/internet_test.jpg"
OUTPUT_DIR = "outputs/"
USER_HEIGHT_CM = 175.0 

# PATHS
ROMP_EXE = r"C:\Users\Anuka\.conda\envs\skin_rehab\Scripts\romp.exe"
MODEL_DIR = r"C:\Users\Anuka\.romp"
SMPL_DIR = r"C:\Users\Anuka\.romp" # Use the .romp folder for consistency

# ==========================================
# 2. RUN AI INFERENCE (ROMP)
# ==========================================
print(f"📸 Processing {INPUT_FILE}...")

# Check for files
if not os.path.exists(ROMP_EXE):
    print(f"❌ ROMP executable missing at: {ROMP_EXE}")
    exit()
    
# Construct full file paths
romp_model_file = os.path.join(MODEL_DIR, "ROMP.pkl")
smpl_model_file = os.path.join(SMPL_DIR, "SMPL_NEUTRAL.pkl")

if not os.path.exists(romp_model_file):
    print(f"❌ ROMP.pkl missing at: {romp_model_file}")
    exit()
if not os.path.exists(smpl_model_file):
    print(f"❌ SMPL_NEUTRAL.pkl missing at: {smpl_model_file}")
    # Fallback check for the other folder
    smpl_model_file = r"C:\Users\Anuka\Desktop\7th Sem\EE7802 UGP\SkinRehab_Project\model_data\SMPL_NEUTRAL.pkl"
    if not os.path.exists(smpl_model_file):
         print(f"❌ SMPL_NEUTRAL.pkl also missing at fallback: {smpl_model_file}")
         exit()

# COMMAND WITH FILE PATHS
cmd = [
    ROMP_EXE,
    "--mode=image",
    "--calc_smpl",
    f"-i={INPUT_FILE}",
    f"-o={OUTPUT_DIR}",
    f"--model_path={romp_model_file}",  # Point to ROMP.pkl
    f"--smpl_path={smpl_model_file}"     # Point to SMPL_NEUTRAL.pkl
]

try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError:
    print("❌ Error: The ROMP command failed.")
    exit()

# ==========================================
# 3. EXTRACT DATA
# ==========================================
result_files = glob.glob(os.path.join(OUTPUT_DIR, "*.npz"))
if not result_files:
    print("❌ ERROR: No output data found.")
    exit()

latest_file = max(result_files, key=os.path.getctime)
print(f"📂 Loading data from: {latest_file}")
data = np.load(latest_file, allow_pickle=True)

# Robust dictionary extraction
results = data['results'].item() if hasattr(data['results'], 'item') else data['results']
if isinstance(results, (list, np.ndarray)) and len(results) > 0:
    results = results[0]

# Betas
if 'smpl_betas' in results:
    betas = results['smpl_betas'][0] if len(results['smpl_betas'].shape) > 1 else results['smpl_betas']
elif 'betas' in results:
    betas = results['betas'][0] if len(results['betas'].shape) > 1 else results['betas']
else:
    betas = np.zeros(10)
    print("⚠️ Warning: Using generic body shape.")

# Joints
joints = None
for key in ['joints', 'j3d', 'smpl_joints', 'kps']:
    if key in results:
        temp_joints = results[key]
        if len(temp_joints.shape) == 3:
            joints = temp_joints[0]
        else:
            joints = temp_joints
        break

if joints is None:
    print("⚠️ Warning: No joints found. Using default scale.")
    scale_factor = 1.0
else:
    if joints.shape[1] >= 2:
        head_y = joints[15, 1]
        foot_y = (joints[7, 1] + joints[8, 1]) / 2
        model_height_units = abs(head_y - foot_y)
        scale_factor = USER_HEIGHT_CM / model_height_units if model_height_units > 0 else 1.0
    else:
        scale_factor = 1.0

# Save
profile_path = "user_profile.npy"
np.save(profile_path, {
    'betas': betas,
    'scale': scale_factor,
    'height_cm': USER_HEIGHT_CM
})

print("\n" + "="*40)
print("✅ DIGITAL TWIN CREATED SUCCESSFULLY")
print(f"📂 Saved to: {profile_path}")
print("="*40)
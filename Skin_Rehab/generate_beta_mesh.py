import torch
import smplx
import numpy as np
import trimesh
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "outputs/sprint_theta/sprint_beta_sequence.csv"
OUTPUT_FILE = "outputs/sprint_theta/runner_shape.glb"

# We use the same model paths from your previous digital twin script
MODEL_FOLDER = r"C:\Users\Anuka\model_data"
GENDER = "neutral"

def generate_mesh():
    print("🚀 INITIALIZING 3D MESH GENERATION...")

    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: Shape data not found at {CSV_FILE}")
        return

    # 1. READ AVERAGE BETAS
    print("📊 Loading Shape Parameters...")
    df = pd.read_csv(CSV_FILE)
    avg_betas = df.mean().values
    
    # Convert to PyTorch tensor (Batch size 1, 10 parameters)
    betas_tensor = torch.tensor(avg_betas).float().unsqueeze(0)

    # 2. CREATE NEUTRAL POSE
    # We want a static standing model, so all 72 joint angles must be exactly 0
    pose_tensor = torch.zeros((1, 72)).float()

    # 3. BUILD MESH WITH SMPL
    print("⚙️  Processing through SMPL Model...")
    try:
        smpl = smplx.SMPL(model_path=MODEL_FOLDER, gender=GENDER)
    except Exception as e:
        print(f"❌ Error loading SMPL: {e}")
        return

    # Pass the shape (betas) and the neutral pose (zeros)
    output = smpl(
        betas=betas_tensor, 
        global_orient=pose_tensor[:, :3], 
        body_pose=pose_tensor[:, 3:72], 
        return_verts=True
    )
    
    verts = output.vertices.detach().cpu().numpy()[0]
    faces = smpl.faces

    # 4. EXPORT TO GLB
    print("💾 Exporting to GLB format...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Rotate 180 degrees on X-axis so the model stands upright in standard 3D viewers (like Windows 3D Viewer)
    R_X180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    mesh.vertices = mesh.vertices @ R_X180.T

    try:
        # Export as GLB (Binary glTF)
        mesh.export(OUTPUT_FILE)
        print("-" * 30)
        print(f"✅ SUCCESS! 3D Model Generated.")
        print(f"💾 Saved to: {OUTPUT_FILE}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Error saving GLB: {e}")
        print("💡 Tip: If trimesh fails to save GLB, try changing OUTPUT_FILE extension to .obj")

if __name__ == "__main__":
    generate_mesh()
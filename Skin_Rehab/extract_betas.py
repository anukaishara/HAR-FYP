import numpy as np
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "outputs/sprint_theta/video_results.npz"
OUTPUT_CSV = "outputs/sprint_theta/sprint_beta_sequence.csv"

def extract_betas():
    print(f"🚀 EXTRACTING SHAPE PARAMETERS (BETAS)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File not found at {INPUT_FILE}")
        return

    # 1. LOAD DATA
    data = np.load(INPUT_FILE, allow_pickle=True)
    results = data['results'].item() if hasattr(data['results'], 'item') else data['results']
    
    beta_sequence = []

    # 2. EXTRACT BETAS
    if isinstance(results, dict):
        try:
            sorted_keys = sorted(results.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
        except:
            sorted_keys = sorted(results.keys())

        for key in sorted_keys:
            frame_data = results[key]
            
            # ROMP usually saves this as 'smpl_betas'
            betas = None
            if 'smpl_betas' in frame_data: betas = frame_data['smpl_betas']
            elif 'betas' in frame_data: betas = frame_data['betas']
            elif 'params' in frame_data and 'betas' in frame_data['params']:
                betas = frame_data['params']['betas']

            if betas is not None:
                beta_flat = np.array(betas).flatten()
                # SMPL shape is defined by 10 parameters
                beta_sequence.append(beta_flat[:10])

    elif isinstance(results, (list, np.ndarray)):
        for frame_data in results:
             if 'smpl_betas' in frame_data:
                 beta_sequence.append(np.array(frame_data['smpl_betas']).flatten()[:10])

    if not beta_sequence:
        print("❌ Error: Could not find 'smpl_betas' in the file.")
        return

    # 3. SAVE AND ANALYZE
    df = pd.DataFrame(beta_sequence, columns=[f"beta_{i}" for i in range(10)])
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Calculate the average shape across the video
    avg_betas = df.mean().values
    
    print("-" * 30)
    print(f"✅ BETA EXPORT COMPLETE")
    print(f"   • Rows Saved: {len(df)}")
    print(f"   • Columns:    10 (Shape Parameters)")
    print(f"   • File:       {OUTPUT_CSV}")
    print("-" * 30)
    print("\n📊 AVERAGE SHAPE (Beta 0-9):")
    print(np.round(avg_betas, 4))
    print("\n💡 Tip: Use these average values if you want to generate a static 3D mesh of the runner!")

if __name__ == "__main__":
    extract_betas()
import numpy as np
import pandas as pd
import cv2
import os

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "inputs/sprint.mp4"
DATA_FILE = "outputs/sprint_theta/video_results.npz"
CSV_FILENAME = "outputs/sprint_theta/sprint_sequence_raw.csv"

def analyze_and_export():
    print(f"🚀 DIAGNOSTIC MODE: Analyzing 'sprint'...")
    
    # 1. CHECK THE ACTUAL VIDEO FILE
    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        real_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = real_count / real_fps if real_fps > 0 else 0
        cap.release()
        
        print("-" * 30)
        print(f"🎥 VIDEO FILE TRUTH:")
        print(f"   • Real FPS:       {real_fps:.2f}")
        print(f"   • Total Frames:   {real_count}")
        print(f"   • Duration:       {duration:.2f} seconds")
        print("-" * 30)
    else:
        print(f"⚠️ Warning: Could not find '{VIDEO_PATH}' to verify FPS.")

    # 2. LOAD ROMP DATA
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    data = np.load(DATA_FILE, allow_pickle=True)
    results = data['results'].item() if hasattr(data['results'], 'item') else data['results']
    
    theta_sequence = []

    # 3. EXTRACT RAW FRAMES (No Interpolation)
    if isinstance(results, dict):
        # Sort by frame number (0, 1, 2...)
        # We assume keys are like 'frame_0.jpg', 'frame_12.jpg'
        try:
            sorted_keys = sorted(results.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
        except:
            sorted_keys = sorted(results.keys())

        print(f"📂 AI DATA TRUTH:")
        print(f"   • Frames Captured: {len(sorted_keys)}")
        
        for key in sorted_keys:
            frame_data = results[key]
            
            # Extract Pose
            pose = None
            if 'smpl_thetas' in frame_data: pose = frame_data['smpl_thetas']
            elif 'pose' in frame_data: pose = frame_data['pose']
            elif 'params' in frame_data: pose = frame_data['params']['pose']

            if pose is not None:
                pose_flat = np.array(pose).flatten()
                theta_sequence.append(pose_flat[:72])

    elif isinstance(results, (list, np.ndarray)):
        print(f"📂 AI DATA TRUTH:")
        print(f"   • Frames Captured: {len(results)}")
        for frame_data in results:
             if 'smpl_thetas' in frame_data:
                 theta_sequence.append(np.array(frame_data['smpl_thetas']).flatten()[:72])

    # 4. SAVE RAW CSV
    df = pd.DataFrame(theta_sequence)
    df.to_csv(CSV_FILENAME, index=False, header=False)
    
    print("-" * 30)
    print(f"✅ EXPORT COMPLETE")
    print(f"   • Rows Saved: {len(df)}")
    print(f"   • File:       {CSV_FILENAME}")
    
    # 5. FINAL VERDICT
    if 'real_count' in locals():
        diff = real_count - len(df)
        if diff == 0:
            print("\n✅ PERFECT MATCH: AI detected every single frame.")
        elif diff > 0:
            print(f"\n⚠️ GAP DETECTED: The AI missed {diff} frames.")
            print("   (This happens if the person ran out of view or was blurry)")
        else:
            print("\n❓ ODD: AI has more frames than video? (Likely a counting error)")

if __name__ == "__main__":
    analyze_and_export()
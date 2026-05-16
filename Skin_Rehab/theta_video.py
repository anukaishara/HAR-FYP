import os
import subprocess
import glob
import time
import cv2
import numpy as np
import pandas as pd
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure your video file name is correct here
VIDEO_NAME = "sprint" 
INPUT_VIDEO_PATH = f"inputs/{VIDEO_NAME}.mp4" # or .avi, .mov
OUTPUT_DIR = "outputs/sprint_theta"
CSV_FILENAME = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_sequence.csv")

# PATHS (Using your specific environment paths)
ROMP_EXE = r"C:\Users\Anuka\.conda\envs\skin_rehab\Scripts\romp.exe"
MODEL_DIR = r"C:\Users\Anuka\.romp"

def numeric_sort_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    digits = ''.join(filter(str.isdigit, name))
    return (0, int(digits)) if digits else (1, name)

def load_results_from_npz(path):
    data = np.load(path, allow_pickle=True)
    if 'results' not in data:
        return data

    results = data['results']
    if hasattr(results, 'item'):
        try:
            return results.item()
        except ValueError:
            return results
    return results

def extract_pose_from_frame(frame_data):
    pose = None
    if isinstance(frame_data, dict):
        if 'smpl_thetas' in frame_data:
            pose = frame_data['smpl_thetas']
        elif 'pose' in frame_data:
            pose = frame_data['pose']
        elif 'params' in frame_data:
            params = frame_data['params']
            if isinstance(params, dict) and 'pose' in params:
                pose = params['pose']
            else:
                pose = params

    if pose is None:
        return None

    pose_flat = np.array(pose).flatten()
    if pose_flat.size < 72:
        return None
    return pose_flat[:72]

def run_video_pipeline():
    print(f"🚀 STARTING VIDEO PROCESSING: {VIDEO_NAME}")
    
    # 1. SETUP & CHECKS
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"❌ Error: Video not found at {INPUT_VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        print(f"⚠️ Warning: Could not read frame count from {INPUT_VIDEO_PATH}.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. RUN ROMP AI (Step 1)
    print("\n[Step 1/2] Running AI Inference (This takes time)...")
    
    cmd = [
        ROMP_EXE,
        "--mode=video",
        "--calc_smpl",
        f"-i={INPUT_VIDEO_PATH}",
        f"-o={OUTPUT_DIR}",
        f"--model_path={os.path.join(MODEL_DIR, 'ROMP.pkl')}",
        f"--smpl_path={os.path.join(MODEL_DIR, 'SMPL_NEUTRAL.pkl')}"
    ]

    try:
        # We redirect output to avoid cluttering the terminal, but check for errors
        start_time = time.time()
        subprocess.check_call(cmd)
        end_time = time.time()
        total_inference_time = end_time - start_time
        average_romp_fps = total_frames / total_inference_time if total_inference_time > 0 and total_frames > 0 else 0.0
        print("✅ AI Inference Complete.")
        print("=" * 40)
        print(f"Total ROMP Inference Time: {total_inference_time:.2f} seconds")
        print(f"Average ROMP FPS:          {average_romp_fps:.2f}")
        print("=" * 40)
    except subprocess.CalledProcessError:
        print("❌ Error: ROMP command failed.")
        return

    # 3. EXTRACT THETAS (Step 2)
    print("\n[Step 2/2] Extracting Theta Sequence...")
    
    # ROMP can write either one aggregate file or many numbered per-frame files.
    named_patterns = [
        os.path.join(OUTPUT_DIR, f"*{VIDEO_NAME}*.npz"),
        os.path.join(OUTPUT_DIR, "video_results.npz")
    ]
    result_files = []
    for pattern in named_patterns:
        result_files.extend(glob.glob(pattern))

    if not result_files:
        result_files = glob.glob(os.path.join(OUTPUT_DIR, "*.npz"))
    
    if not result_files:
        print("❌ Error: No output .npz file found.")
        return

    theta_sequence = []

    aggregate_files = [
        path for path in result_files
        if os.path.basename(path) == "video_results.npz" or VIDEO_NAME in os.path.basename(path)
    ]

    if aggregate_files:
        target_file = max(aggregate_files, key=os.path.getctime)
        print(f"📂 Loading aggregate data from: {os.path.basename(target_file)}")
        results = load_results_from_npz(target_file)

        if isinstance(results, dict):
            flat_pose = extract_pose_from_frame(results)
            if flat_pose is not None:
                theta_sequence.append(flat_pose)
            else:
                sorted_keys = sorted(results.keys(), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
                print(f"   -> Found {len(sorted_keys)} frames.")

                for key in sorted_keys:
                    pose = extract_pose_from_frame(results[key])
                    if pose is not None:
                        theta_sequence.append(pose)

        elif isinstance(results, (list, np.ndarray)):
            print(f"   -> Found list with {len(results)} frames.")
            for frame_data in results:
                pose = extract_pose_from_frame(frame_data)
                if pose is not None:
                    theta_sequence.append(pose)
    else:
        frame_files = sorted(result_files, key=numeric_sort_key)
        print(f"📂 Loading {len(frame_files)} per-frame NPZ files.")

        for frame_file in frame_files:
            results = load_results_from_npz(frame_file)
            pose = extract_pose_from_frame(results)

            if pose is None and isinstance(results, dict):
                for frame_data in results.values():
                    pose = extract_pose_from_frame(frame_data)
                    if pose is not None:
                        break

            if pose is not None:
                theta_sequence.append(pose)

    # 4. SAVE TO CSV
    if len(theta_sequence) == 0:
        print("❌ Error: No pose data extracted.")
        return

    df = pd.DataFrame(theta_sequence)
    df.to_csv(CSV_FILENAME, index=False, header=False)
    
    print("-" * 30)
    print(f"✅ SUCCESS!")
    print(f"📊 Extracted {len(df)} frames.")
    print(f"💾 Saved to: {CSV_FILENAME}")
    print("-" * 30)

if __name__ == "__main__":
    run_video_pipeline()

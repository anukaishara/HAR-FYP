import json
import os

import numpy as np


INPUT_CSV = "outputs/sprint_romp_thetas.csv"
OUTPUT_JSON = "outputs/sprint_theta/romp_metrics.json"

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shldr", "R_Shldr",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


def main():
    theta = np.loadtxt(INPUT_CSV, delimiter=",")

    if theta.ndim == 1:
        theta = theta.reshape(1, -1)

    if theta.shape[1] != 72:
        raise ValueError(f"Expected theta data with 72 columns, got shape {theta.shape}")

    frame_count = int(theta.shape[0])
    if frame_count < 2:
        raise ValueError("Need at least 2 frames to compute frame-to-frame jitter")

    jitter = np.mean(np.abs(theta[1:] - theta[:-1]), axis=0)
    per_joint_jitter = jitter.reshape(24, 3).mean(axis=1)

    metrics = {
        "frame_count": frame_count,
        "mean_jitter": float(jitter.mean()),
        "median_jitter": float(np.median(jitter)),
        "p75_jitter": float(np.percentile(jitter, 75)),
        "p95_jitter": float(np.percentile(jitter, 95)),
        "max_jitter": float(jitter.max()),
        "mean_variance": float(np.var(theta, axis=0).mean()),
        "smoothness_score": float(1 / (1 + jitter.mean())),
        "per_channel_jitter": jitter.tolist(),
        "per_joint_jitter": per_joint_jitter.tolist(),
        "joint_names": JOINT_NAMES,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("ROMP Sprint Theta Metrics")
    print("=" * 32)
    print(f"Frame Count:      {metrics['frame_count']}")
    print(f"Mean Jitter:      {metrics['mean_jitter']:.10f}")
    print(f"Median Jitter:    {metrics['median_jitter']:.10f}")
    print(f"P75 Jitter:       {metrics['p75_jitter']:.10f}")
    print(f"P95 Jitter:       {metrics['p95_jitter']:.10f}")
    print(f"Max Jitter:       {metrics['max_jitter']:.10f}")
    print(f"Mean Variance:    {metrics['mean_variance']:.10f}")
    print(f"Smoothness Score: {metrics['smoothness_score']:.10f}")
    print("=" * 32)
    print(f"Saved JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

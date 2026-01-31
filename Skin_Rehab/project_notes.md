# Project Context: SkinRehab Digital Twin Engine

**Current State:** Optimized, 60 FPS, Hardware-Accelerated Animation Player.
**Hardware:** Ryzen 7 5800H | RTX 3050 (4GB) | Windows 11
**Objective:** Visualize biomechanical motion data (SMPL/ROMP) on a personalized 3D avatar with high engineering precision.

---

## 1. Technical Architecture

### **Core Libraries**

* **PyTorch:** Matrix operations and tensor handling for pose data.
* **SMPL-X:** Body model layer (Skinning, Shape Betas, Joint Rotations).
* **Trimesh:** 3D rendering and geometry management.
* **Numpy:** Data interpolation and array manipulation.

### **Data Pipeline**

1. **Input:** Single image -> `ROMP` (ResNet-50) -> Generates `.npz` file containing pose (`thetas`) and shape (`betas`).
2. **Processing:**
* **Universal Hunter:** Recursively scans `.npz` files to find pose arrays (Shape `N, 72`), handling both nested dictionaries and flat lists.
* **Stitching:** Automatically detects if data is a "Video Block" (continuous) or "Frame Sequence" (discrete) and stitches them.
* **Interpolation:** Upsamples 30 FPS raw data to 60 FPS using linear interpolation to prevent visual jitter ("teleporting").



### **Physics Engine (The "Pre-Bake" System)**

To achieve 60+ FPS on Python, real-time calculation was replaced with a pre-bake step:

1. **Warm-Up:** The script runs the SMPL forward pass for *all frames* at startup (taking ~1-2s).
2. **Transformation:**
* **Flip:** Applies a  X-axis rotation matrix to fix coordinate mismatch (Computer Vision Y-Down  3D Graphics Y-Up).
* **Lift:** Applies a Y-axis offset (+1.05m) to correct floor penetration.


3. **Playback:** The rendering loop performs **zero math**. It simply swaps pre-calculated vertex arrays from memory to GPU.

---

## 2. Solved Engineering Challenges

| Issue | Cause | Solution |
| --- | --- | --- |
| **"Moon Gravity"** | Python `time.sleep()` is imprecise and CPU scheduling caused lag. | Replaced with a **Busy Wait Loop** that spins the CPU to hit exact millisecond targets. |
| **"Teleporting"** | Skipping frames to maintain sync caused large jumps in motion. | Implemented **Linear Interpolation** to generate intermediate frames (30  60 FPS). |
| **Low GPU Usage** | Windows defaulted `python.exe` to Integrated Graphics. | Hard-forced **High Performance (RTX 3050)** in Windows Graphics Settings. |
| **Crash on Run** | ROMP data structure varies (dict vs list) between versions. | Wrote a **recursive search algorithm** to find the `72`-float pose array anywhere in the file. |

---
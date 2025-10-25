"""
depth_mask_strategies.py
------------------------
Utility functions to create masks from depth maps using different
thresholding strategies (relative, percentile, Otsu, calibrated).
"""

import os
import numpy as np
import cv2

# ---------------------------------------------------------------------
# Base normalization helper
# ---------------------------------------------------------------------
def normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

# ---------------------------------------------------------------------
# 1. Fixed normalized threshold
# ---------------------------------------------------------------------
def mask_fixed(depth: np.ndarray, near_thresh=0.3, far_thresh=None):
    d = normalize_depth(depth)
    if far_thresh is None:
        mask = d < near_thresh
    else:
        mask = (d > near_thresh) & (d < far_thresh)
    return (mask.astype(np.uint8) * 255)

# ---------------------------------------------------------------------
# 2. Percentile-based threshold
# ---------------------------------------------------------------------
def mask_percentile(depth: np.ndarray, lower_pct=10, upper_pct=None):
    low = np.percentile(depth, lower_pct)
    if upper_pct is None:
        mask = depth < low
    else:
        high = np.percentile(depth, upper_pct)
        mask = (depth > low) & (depth < high)
    return (mask.astype(np.uint8) * 255)

# ---------------------------------------------------------------------
# 3. Otsu automatic threshold
# ---------------------------------------------------------------------
def mask_otsu(depth: np.ndarray, keep='near', bias: float = 0.0):
    d = normalize_depth(depth)
    d_8u = (d * 255).astype(np.uint8)
    t_otsu = cv2.threshold(d_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    t_shifted = np.clip(t_otsu + bias, 0.0, 1.0)
    t_8u = int(t_shifted * 255)
    _, mask = cv2.threshold(d_8u, t_8u, 255, cv2.THRESH_BINARY)
    if keep == 'near':
        mask = 255 - mask
    elif keep != 'far':
        raise ValueError("keep must be 'near' or 'far'")
    return mask

# ---------------------------------------------------------------------
# 4. Calibrated metric band (approximate)
# ---------------------------------------------------------------------
def mask_calibrated(depth: np.ndarray, scale: float, near_m=0.6, far_m=0.9):
    depth_m = depth * scale
    if np.mean(depth_m[:50, :50]) > np.mean(depth_m[-50:, -50:]):  # near is brighter
        mask = (depth_m < near_m) & (depth_m > far_m)
    else:
        mask = (depth_m > near_m) & (depth_m < far_m)
    return (mask.astype(np.uint8) * 255)

# ---------------------------------------------------------------------
# Directory batch processing (NEW)
# ---------------------------------------------------------------------
def process_depth_dir(input_dir="./outputs", output_dir="./mask_path"):
    os.makedirs(output_dir, exist_ok=True)
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Processing {len(npy_files)} depth maps in {input_dir} ...")

    # choose method (as before)
    method_name = "percentile"  # change this section below if desired

    # parameters tuned by your previous calibration
    if method_name == "calibrated":
        name = "calibrated"
        near_m = 0
        far_m = 0.90
        for fname in npy_files:
            depth = np.load(os.path.join(input_dir, fname))
            mask = mask_calibrated(depth, scale=0.00045, near_m=near_m, far_m=far_m)
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_{name}.png"), mask)

    elif method_name == "percentile":
        name = "percentile"
        lower_pct = 1
        for fname in npy_files:
            depth = np.load(os.path.join(input_dir, fname))
            mask = mask_percentile(depth, lower_pct=lower_pct)
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_{name}.png"), mask)

    elif method_name == "fixed":
        name = "fixed"
        near_thresh = 0.85
        for fname in npy_files:
            depth = np.load(os.path.join(input_dir, fname))
            mask = mask_fixed(depth, near_thresh=near_thresh)
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_{name}.png"), mask)

    elif method_name == "otsu":
        name = "otsu"
        bias = 0.3
        for fname in npy_files:
            depth = np.load(os.path.join(input_dir, fname))
            mask = mask_otsu(depth, keep='near', bias=bias)
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_{name}.png"), mask)

    print(f"Saved {len(npy_files)} masks to {output_dir}")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    process_depth_dir(input_dir="./images/positives", output_dir="./mask_path/positives")
    process_depth_dir(input_dir="./images/negatives", output_dir="./mask_path/negatives")

"""
depth_mask_strategies.py
------------------------
Utility functions to create masks from depth maps using different
thresholding strategies (relative, percentile, Otsu, calibrated).
"""

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
    """
    Keeps pixels below a fixed normalized depth threshold.
    """
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
    """
    Keeps pixels within certain percentiles of depth values.
    """
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
def mask_otsu(depth: np.ndarray):
    """
    Uses Otsu's method to automatically separate near/far regions.
    """
    d = normalize_depth(depth)
    d_8u = (d * 255).astype(np.uint8)
    _, mask = cv2.threshold(d_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - mask  # invert so near=white


# ---------------------------------------------------------------------
# 4. Calibrated metric band (approximate)
# ---------------------------------------------------------------------
def mask_calibrated(depth: np.ndarray, scale: float, near_m=0.6, far_m=0.9):
    """
    Keeps pixels whose (scaled) depth is within a metric range.
    scale: meters per model-unit (computed via calibration)
    """
    depth_m = depth * scale
    mask = (depth_m > near_m) & (depth_m < far_m)
    return (mask.astype(np.uint8) * 255)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    depth = np.load("./outputs/e_raw.npy")

    # choose method
    # mask = mask_fixed(depth, near_thresh=0.65) #[0.65 0.75] # e
    # mask = mask_percentile(depth, lower_pct=55) # [55] # The worst one imo
    mask = mask_otsu(depth)
    # mask = mask_calibrated(depth, scale=0.00045, near_m=0.0, far_m=0.7) # 0.5, near = lower far = upper
# 1 – 2 ft	near_m=0.3, far_m=0.6 # 
# 2 – 3 ft	near_m=0.6, far_m=0.9
# 3 – 5 ft	near_m=0.9, far_m=1.5
    cv2.imwrite("./mask_path/mask_output.png", mask)
    print("Saved mask_output.png")

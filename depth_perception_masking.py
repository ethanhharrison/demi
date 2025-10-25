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
def mask_otsu(depth: np.ndarray, keep='near', bias: float = 0.0):
    """
    Otsu's method with a bias shift in normalized depth units [0, 1].

    Args:
        depth: Raw depth map (float32 array).
        keep:  'near' or 'far' — which side of the threshold to keep.
        bias:  Float in range [-1, 1]. Positive bias shifts threshold toward farther pixels
               (includes more near region if keep='near').

    Returns:
        Binary mask (uint8, values 0 or 255).
    """
    d = normalize_depth(depth)
    d_8u = (d * 255).astype(np.uint8)

    # Get Otsu threshold in normalized units
    t_otsu = cv2.threshold(d_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0

    # Apply bias in normalized units, then convert back to 8-bit
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
    """
    Keeps pixels whose (scaled) depth is within a metric range.
    Auto-handles inverted depth (larger = nearer).
    """
    depth_m = depth * scale
    # Detect inversion
    if np.mean(depth_m[:50, :50]) > np.mean(depth_m[-50:, -50:]):  # near is brighter
        # inverted encoding
        mask = (depth_m < near_m) & (depth_m > far_m)
    else:
        # normal encoding
        mask = (depth_m > near_m) & (depth_m < far_m)
    return (mask.astype(np.uint8) * 255)



# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    depth1 = np.load("./outputs/first_raw.npy")
    depth2 = np.load("./outputs/second_raw.npy")
    depth3 = np.load("./outputs/third_raw.npy")
    depth4 = np.load("./outputs/fourth_raw.npy")

    # choose method
    # mask = mask_fixed(depth, near_thresh=0.65) #[0.65 0.75] # e
    # mask = mask_percentile(depth, lower_pct=55) # [55] # The worst one imo
    if False:
        name = 'calibrated' # second
        near_m = 0
        far_m = 0.90
        mask1 = mask_calibrated(depth1, scale=0.00045, near_m=near_m, far_m=far_m)
        mask2 = mask_calibrated(depth2, scale=0.00045, near_m=near_m, far_m=far_m)
        mask3 = mask_calibrated(depth3, scale=0.00045, near_m=near_m, far_m=far_m)
        mask4 = mask_calibrated(depth4, scale=0.00045, near_m=near_m, far_m=far_m)
    if True: # 4th
        name = 'percentile'
        lower_pct = 1
        mask1 = mask_percentile(depth1, lower_pct=lower_pct)
        mask2 = mask_percentile(depth2, lower_pct=lower_pct)
        mask3 = mask_percentile(depth3, lower_pct=lower_pct)
        mask4 = mask_percentile(depth4, lower_pct=lower_pct)
    if False: # 0.85
        near_thresh = 0.85
        name = 'fixed'
        mask1 = mask_fixed(depth1, near_thresh=near_thresh)
        mask2 = mask_fixed(depth2, near_thresh=near_thresh)
        mask3 = mask_fixed(depth3, near_thresh=near_thresh)
        mask4 = mask_fixed(depth4, near_thresh=near_thresh)
    if False:
        bias = 0.3 # negative bias increase distance
        name = 'otsu'
        mask1 = mask_otsu(depth1, keep='near', bias=bias)
        mask2 = mask_otsu(depth2, keep='near', bias=bias)
        mask3 = mask_otsu(depth3, keep='near', bias=bias)
        mask4 =mask_otsu(depth4, keep='near', bias=bias)
    # mask = mask_calibrated(depth, scale=0.00045, near_m=0.0, far_m=0.7) # 0.5, near = lower far = upper
# 1 – 2 ft	near_m=0.3, far_m=0.6 # 
# 2 – 3 ft	near_m=0.6, far_m=0.9
# 3 – 5 ft	near_m=0.9, far_m=1.5
    cv2.imwrite(f"./mask_path/mask_output1_{name}.png", mask1)
    cv2.imwrite(f"./mask_path/mask_output2_{name}.png", mask2)
    cv2.imwrite(f"./mask_path/mask_output3_{name}.png", mask3)
    cv2.imwrite(f"./mask_path/mask_output4_{name}.png", mask4)
    print("Saved mask_output.png")

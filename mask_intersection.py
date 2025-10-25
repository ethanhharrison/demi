import numpy as np

import numpy as np

def mask_overlap_percentage(
    mask_depth: np.ndarray,
    mask_seg: np.ndarray,
    invert_depth: bool = False,
    invert_seg: bool = False,
) -> float:
    """
    Compute the percentage overlap (IoU × 100) between two binary masks,
    verifying that they have the same shape, dtype, and valid binary values.
    Optionally invert either mask so that black or white can represent the
    "kept" region.

    Args:
        mask_depth (np.ndarray): Binary depth mask (0 or 1).
        mask_seg (np.ndarray): Binary segmentation mask (0 or 1).
        invert_depth (bool): If True, invert depth mask (swap 0↔1).
        invert_seg (bool): If True, invert segmentation mask (swap 0↔1).

    Returns:
        float: Overlap percentage between 0 and 100.
    """
    # --- Validate shapes ---
    if mask_depth.shape != mask_seg.shape:
        raise ValueError(f"Mask shape mismatch: {mask_depth.shape} vs {mask_seg.shape}")

    # --- Validate dtypes ---
    if mask_depth.dtype != mask_seg.dtype:
        raise ValueError(f"Mask dtype mismatch: {mask_depth.dtype} vs {mask_seg.dtype}")

    # --- Validate binary format ---
    for name, mask in [("Depth", mask_depth), ("Segmentation", mask_seg)]:
        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(f"{name} mask has non-binary values: {unique_vals}")

    # --- Invert if requested ---
    if invert_depth:
        mask_depth = 1 - mask_depth
    if invert_seg:
        mask_seg = 1 - mask_seg

    # --- Compute IoU × 100 ---
    intersection = np.logical_and(mask_depth, mask_seg).sum()
    union = np.logical_or(mask_depth, mask_seg).sum()

    if union == 0:
        return 0.0  # both empty or invalid

    return float(intersection / union * 100)




import cv2
import numpy as np
from typing import Union

import cv2
import numpy as np
from typing import Union

def mask_overlap_from_paths(
    depth_path: Union[str, bytes],
    seg_path: Union[str, bytes],
    threshold: int = 127,
    invert_depth: bool = False,
    invert_seg: bool = False,
) -> float:
    """
    Load two mask images from file paths and compute their overlap percentage (IoU × 100).

    Args:
        depth_path (str): Path to the binary depth mask image.
        seg_path (str): Path to the binary segmentation mask image.
        threshold (int): Pixel intensity threshold (0–255) used to binarize grayscale masks.
        invert_depth (bool): If True, invert depth mask (swap 0↔1).
        invert_seg (bool): If True, invert segmentation mask (swap 0↔1).

    Returns:
        float: Overlap percentage between 0 and 100.
    """
    # --- Load masks ---
    mask_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    mask_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

    if mask_depth is None:
        raise FileNotFoundError(f"Could not read depth mask from: {depth_path}")
    if mask_seg is None:
        raise FileNotFoundError(f"Could not read segmentation mask from: {seg_path}")

    # --- Binarize ---
    mask_depth = (mask_depth > threshold).astype(np.uint8)
    mask_seg = (mask_seg > threshold).astype(np.uint8)

    # --- Invert if needed ---
    if invert_depth:
        mask_depth = 1 - mask_depth
    if invert_seg:
        mask_seg = 1 - mask_seg

    # --- Validate shape ---
    if mask_depth.shape != mask_seg.shape:
        raise ValueError(f"Mask shape mismatch: {mask_depth.shape} vs {mask_seg.shape}")

    # --- Ensure binary values ---
    for name, mask in [("Depth", mask_depth), ("Segmentation", mask_seg)]:
        vals = np.unique(mask)
        if not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"{name} mask has non-binary values: {vals}")

    # --- Compute IoU × 100 ---
    intersection = np.logical_and(mask_depth, mask_seg).sum()
    union = np.logical_or(mask_depth, mask_seg).sum()

    if union == 0:
        return 0.0

    return float(intersection / union * 100)


import numpy as np
import cv2
import tempfile
import os

# --- assume mask_overlap_from_paths() is already defined above ---

def create_dummy_mask(shape, fill_coords=None):
    """Helper: create a binary mask with optional filled rectangle coords."""
    mask = np.zeros(shape, dtype=np.uint8)
    if fill_coords:
        x0, y0, x1, y1 = fill_coords
        mask[y0:y1, x0:x1] = 1
    return mask


def save_temp_mask(mask):
    """Save a mask to a temporary PNG and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp.name, mask * 255)  # scale 0/1 to 0/255 for PNG
    return tmp.name


def run_dummy_tests():
    print("Running dummy mask overlap tests...\n")

    # --- Case 1: identical masks (100% overlap) ---
    mask1 = create_dummy_mask((100, 100), (20, 20, 80, 80))
    mask2 = mask1.copy()
    p1 = save_temp_mask(mask1)
    p2 = save_temp_mask(mask2)
    print("Test 1 (identical):", mask_overlap_from_paths(p1, p2))

    # --- Case 2: partial overlap (25%) ---
    mask1 = create_dummy_mask((100, 100), (20, 20, 80, 80))
    mask2 = create_dummy_mask((100, 100), (50, 50, 90, 90))
    p1 = save_temp_mask(mask1)
    p2 = save_temp_mask(mask2)
    print("Test 2 (partial overlap):", mask_overlap_from_paths(p1, p2))

    # --- Case 3: no overlap (0%) ---
    mask1 = create_dummy_mask((100, 100), (10, 10, 40, 40))
    mask2 = create_dummy_mask((100, 100), (60, 60, 90, 90))
    p1 = save_temp_mask(mask1)
    p2 = save_temp_mask(mask2)
    print("Test 3 (no overlap):", mask_overlap_from_paths(p1, p2))

    # --- Case 4: mismatched shapes (should raise ValueError) ---
    try:
        mask1 = create_dummy_mask((100, 100), (10, 10, 40, 40))
        mask2 = create_dummy_mask((80, 80), (10, 10, 40, 40))
        p1 = save_temp_mask(mask1)
        p2 = save_temp_mask(mask2)
        print("Test 4 (mismatched shapes):", mask_overlap_from_paths(p1, p2))
    except ValueError as e:
        print("Test 4 (expected error):", e)

    # --- Case 5: empty masks (both zeros → overlap = 0) ---
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    p1 = save_temp_mask(mask1)
    p2 = save_temp_mask(mask2)
    print("Test 5 (both empty):", mask_overlap_from_paths(p1, p2))

    print("\nAll tests complete.")

    # cleanup
    for f in os.listdir(tempfile.gettempdir()):
        if f.endswith(".png"):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), f))
            except:
                pass

    

if __name__ == '__main__':
    overlap = mask_overlap_from_paths(
    "mask_path/mask_output1_percentile.png",
    "mask_path/mask_output4_fixed.png",
    invert_depth=True,
    invert_seg=True
    )
    print(f"Mask overlap: {overlap:.2f}%")
    # run_dummy_tests()

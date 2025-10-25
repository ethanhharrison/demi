import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Tuple, Optional, Union
import os
import cv2
import csv
import tempfile
import numpy as np
from typing import Union, List, Tuple

def load_and_normalize_image(
    image_path: Union[str, bytes],
    target_size: Optional[Tuple[int, int]] = None,
    as_numpy: bool = True,
) -> Union[np.ndarray, Image.Image]:
    """
    Load an image from path (any common format), fix orientation, convert to RGB,
    normalize to [0,1] float32, and optionally resize.

    Args:
        image_path: Path to the image file (.jpg, .png, .bmp, .tiff, etc.).
        target_size: Optional (width, height) to resize the image. If None, keep original size.
        as_numpy: If True, return a NumPy array (H,W,3) normalized to [0,1].
                  If False, return a PIL.Image in RGB format.

    Returns:
        np.ndarray or PIL.Image: Normalized image (float32 if array, else PIL Image).
    """
    # --- Load with PIL (handles most formats robustly) ---
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    # --- Resize if requested ---
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)

    if not as_numpy:
        return img

    # --- Convert to NumPy + normalize to [0,1] ---
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr




# ---------------------------------------------------------------------
# Core: Array-to-array overlap
# ---------------------------------------------------------------------
def mask_overlap_percentage(
    mask_depth: np.ndarray,
    mask_seg: np.ndarray,
    invert_depth: bool = True,
    invert_seg: bool = False,
) -> float:
    """
    Compute IoU x 100 between two binary masks (values ∈ {0,1}).

    Args:
        mask_depth: Binary depth mask.
        mask_seg: Binary segmentation mask.
        invert_depth: Invert depth mask (swap 0↔1).
        invert_seg: Invert segmentation mask (swap 0↔1).

    Returns:
        Overlap percentage (0-100).
    """
    if mask_depth.shape != mask_seg.shape:
        raise ValueError(f"Mask shape mismatch: {mask_depth.shape} vs {mask_seg.shape}")
    if mask_depth.dtype != mask_seg.dtype:
        raise ValueError(f"Mask dtype mismatch: {mask_depth.dtype} vs {mask_seg.dtype}")

    for name, mask in [("Depth", mask_depth), ("Segmentation", mask_seg)]:
        vals = np.unique(mask)
        if not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"{name} mask has non-binary values: {vals}")

    if invert_depth:
        mask_depth = 1 - mask_depth
    if invert_seg:
        mask_seg = 1 - mask_seg

    intersection = np.logical_and(mask_depth, mask_seg).sum()
    union = np.logical_or(mask_depth, mask_seg).sum()
    return float(intersection / union * 100) if union > 0 else 0.0


# ---------------------------------------------------------------------
# Convenience: Path-to-path comparison
# ---------------------------------------------------------------------
def mask_overlap_from_paths(
    depth_path: Union[str, bytes],
    seg_path: Union[str, bytes],
    threshold: int = 127,
    invert_depth: bool = True,
    invert_seg: bool = False,
) -> float:
    """
    Load two grayscale mask images and compute IoU × 100.

    Args:
        depth_path: Path to binary depth mask (.png).
        seg_path: Path to binary segmentation mask (.png).
        threshold: Intensity threshold for binarization.
        invert_depth: Invert depth mask.
        invert_seg: Invert segmentation mask.

    Returns:
        Overlap percentage (0–100).
    """
    mask_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    mask_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

    if mask_depth is None:
        raise FileNotFoundError(f"Could not read depth mask from: {depth_path}")
    if mask_seg is None:
        raise FileNotFoundError(f"Could not read segmentation mask from: {seg_path}")

    mask_depth = (mask_depth > threshold).astype(np.uint8)
    mask_seg = (mask_seg > threshold).astype(np.uint8)

    return mask_overlap_percentage(mask_depth, mask_seg, invert_depth, invert_seg)


# ---------------------------------------------------------------------
# Folder-to-folder batch comparison
# ---------------------------------------------------------------------
def compare_mask_folders(
    folder_depth: str,
    folder_seg: str,
    threshold: int = 127,
    invert_depth: bool = False,
    invert_seg: bool = False,
    output_csv: str = "./mask_overlap_results.csv",
) -> List[Tuple[str, str, float]]:
    """
    Compare masks in two folders (sorted by filename) and compute IoU × 100.

    Args:
        folder_depth: Folder containing depth masks (.png).
        folder_seg: Folder containing segmentation masks (.png).
        threshold: Binarization threshold (0–255).
        invert_depth: Invert depth mask.
        invert_seg: Invert segmentation mask.
        output_csv: Optional path to save CSV results.

    Returns:
        List of tuples: (depth_filename, seg_filename, overlap%)
    """
    depth_files = sorted([f for f in os.listdir(folder_depth) if f.endswith(".png")])
    seg_files = sorted([f for f in os.listdir(folder_seg) if f.endswith(".png")])

    n = min(len(depth_files), len(seg_files))
    if n == 0:
        raise ValueError("No overlapping image pairs found between folders.")

    print(f"Comparing {n} pairs from:\n  {folder_depth}\n  {folder_seg}\n")
    results = []

    for i in range(n):
        dfile, sfile = depth_files[i], seg_files[i]
        dpath, spath = os.path.join(folder_depth, dfile), os.path.join(folder_seg, sfile)

        try:
            overlap = mask_overlap_from_paths(
                dpath, spath,
                threshold=threshold,
                invert_depth=invert_depth,
                invert_seg=invert_seg,
            )
            print(f"[{i+1}/{n}] {dfile} vs {sfile} → {overlap:.2f}%")
            results.append((dfile, sfile, overlap))
        except Exception as e:
            print(f"[{i+1}/{n}] Error comparing {dfile} and {sfile}: {e}")

    # Optional CSV output
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["depth_file", "seg_file", "overlap_percent"])
            writer.writerows(results)
        print(f"\n[Saved] Results written to {output_csv}")

    return results


# ---------------------------------------------------------------------
# Dummy test utilities
# ---------------------------------------------------------------------
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
    cv2.imwrite(tmp.name, mask * 255)
    return tmp.name


def rename_files_in_order(folder_path: str):
    """
    Renames all files in the given folder sequentially: 1.ext, 2.ext, ...
    """
    files = sorted(
        [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    )

    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{i}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")




import numpy as np

def dominant_quadrant(mask: np.ndarray) -> int:
    """
    Divide a 2D mask into 9 equal quadrants (3x3)
    and find which quadrant contains the centroid
    (center of mass) of all nonzero pixels.

    Args:
        mask (np.ndarray): 2D binary or grayscale mask.

    Returns:
        int: Quadrant number (1–9), row-wise:
              1 2 3
              4 5 6
              7 8 9
             Returns -1 if mask has no nonzero pixels.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    # Get coordinates of all nonzero points
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return -1  # empty mask

    # Compute centroid (center of mass)
    cx, cy = np.mean(xs), np.mean(ys)

    # Image dimensions and step size
    h, w = mask.shape
    h_step, w_step = h / 3.0, w / 3.0

    # Determine which row and column the centroid falls into
    row = int(np.clip(cy // h_step, 0, 2))
    col = int(np.clip(cx // w_step, 0, 2))

    # Convert (row, col) into quadrant number (1–9)
    quadrant = row * 3 + col + 1
    return quadrant



# ---------------------------------------------------------------------
# Example entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Positives")
    compare_mask_folders(
        folder_depth="./mask_path/positives",
        folder_seg="./segmentation_images/positives",
        invert_depth=True,
        invert_seg=False,
    )

    print("\nNegatives")
    compare_mask_folders(
        folder_depth="./mask_path/negatives",
        folder_seg="./segmentation_images/negatives",
        invert_depth=True,
        invert_seg=False,
    )
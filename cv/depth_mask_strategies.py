"""
depth_mask_generator.py
------------------------
Encapsulates all depth-mask creation strategies (fixed, percentile, Otsu, calibrated)
into a unified class.

Default: method="fixed" with near_thresh=0.7 (larger → farther)
"""

import os
import numpy as np
import cv2


class DepthMaskGenerator:
    def __init__(
        self,
        method: str = "fixed",
        near_thresh: float = 0.7,       # fixed threshold default
        far_thresh: float = None,
        lower_pct: float = 1.0,         # percentile
        upper_pct: float = None,
        bias: float = 0.3,              # otsu
        keep: str = "near",
        scale: float = 0.00045,         # calibrated
        near_m: float = 0.0,
        far_m: float = 0.9,
    ):
        valid_methods = {"fixed", "percentile", "otsu", "calibrated"}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}', must be one of {valid_methods}")

        self.method = method
        self.near_thresh = near_thresh
        self.far_thresh = far_thresh
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.bias = bias
        self.keep = keep
        self.scale = scale
        self.near_m = near_m
        self.far_m = far_m

    # Normalization helper
    @staticmethod
    def normalize_depth(depth: np.ndarray) -> np.ndarray:
        """Normalize a raw depth array to [0, 1]."""
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Mask generation methods
    def mask_fixed(self, depth: np.ndarray) -> np.ndarray:
        d = self.normalize_depth(depth)
        if self.far_thresh is None:
            mask = d < self.near_thresh
        else:
            mask = (d > self.near_thresh) & (d < self.far_thresh)
        return (mask.astype(np.uint8) * 255)

    def mask_percentile(self, depth: np.ndarray) -> np.ndarray:
        low = np.percentile(depth, self.lower_pct)
        if self.upper_pct is None:
            mask = depth < low
        else:
            high = np.percentile(depth, self.upper_pct)
            mask = (depth > low) & (depth < high)
        return (mask.astype(np.uint8) * 255)

    def mask_otsu(self, depth: np.ndarray) -> np.ndarray:
        d = self.normalize_depth(depth)
        d_8u = (d * 255).astype(np.uint8)
        t_otsu = cv2.threshold(d_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
        t_shifted = np.clip(t_otsu + self.bias, 0.0, 1.0)
        t_8u = int(t_shifted * 255)
        _, mask = cv2.threshold(d_8u, t_8u, 255, cv2.THRESH_BINARY)
        if self.keep == "near":
            mask = 255 - mask
        elif self.keep != "far":
            raise ValueError("keep must be 'near' or 'far'")
        return mask

    def mask_calibrated(self, depth: np.ndarray) -> np.ndarray:
        depth_m = depth * self.scale
        if np.mean(depth_m[:50, :50]) > np.mean(depth_m[-50:, -50:]):  # near brighter
            mask = (depth_m < self.near_m) & (depth_m > self.far_m)
        else:
            mask = (depth_m > self.near_m) & (depth_m < self.far_m)
        return (mask.astype(np.uint8) * 255)

    # Dispatcher
    def generate(self, depth: np.ndarray) -> np.ndarray:
        """Apply the selected masking strategy."""
        if self.method == "fixed":
            return self.mask_fixed(depth)
        elif self.method == "percentile":
            return self.mask_percentile(depth)
        elif self.method == "otsu":
            return self.mask_otsu(depth)
        elif self.method == "calibrated":
            return self.mask_calibrated(depth)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # Directory batch processor
    def process_dir(self, input_dir="./outputs", output_dir="./mask_path"):
        """Generate masks for all .npy files in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
        if not npy_files:
            print(f"No .npy files found in {input_dir}")
            return

        print(f"[{self.method.upper()}] Processing {len(npy_files)} depth maps in {input_dir}...")

        for fname in npy_files:
            depth = np.load(os.path.join(input_dir, fname))
            mask = self.generate(depth)
            out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_{self.method}.png")
            cv2.imwrite(out_path, mask)

        print(f"Saved {len(npy_files)} masks to {output_dir}")



if __name__ == "__main__":
    # Default (fixed) method — no args needed
    generator = DepthMaskGenerator()

    # Single file
    depth = np.load("./outputs/positives/sample_raw.npy")
    mask = generator.generate(depth)
    cv2.imwrite("mask_fixed.png", mask)

    # Batch processing
    generator.process_dir("./outputs/positives", "./mask_path/positives")

"""
depth_estimator.py
------------------
Encapsulates Intel's DPT depth estimation model (tiny 256 variant)
for single-image inference or NumPy array input.

Default model: Intel/dpt-swinv2-tiny-256
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from transformers import AutoModelForDepthEstimation, AutoImageProcessor


class DepthEstimator:
    def __init__(
        self,
        model_id: str = "Intel/dpt-swinv2-tiny-256",
        model_dir: str = "./depth_model",
        device: str = None,
    ):
        """
        Initialize the depth estimator and load (or download) the model.
        """
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"[Init] DepthEstimator using {self.model_id} on {self.device}")
        self.model, self.processor = self._load_or_download()

    # ------------------------------------------------------------------
    # Private model loader
    # ------------------------------------------------------------------
    def _load_or_download(self):
        """Load local model if available, otherwise download and cache."""
        if os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"[Load] Found local model at {self.model_dir}")
            model = AutoModelForDepthEstimation.from_pretrained(self.model_dir)
            processor = AutoImageProcessor.from_pretrained(self.model_dir, use_fast=True)
        else:
            print(f"[Setup] Downloading model weights from {self.model_id}...")
            os.makedirs(self.model_dir, exist_ok=True)
            model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            model.save_pretrained(self.model_dir)
            processor.save_pretrained(self.model_dir)
            print(f"[Done] Model + processor saved at {self.model_dir}")

        return model.to(self.device).eval(), processor

    # ------------------------------------------------------------------
    # Public: single image or array inference
    # ------------------------------------------------------------------
    def infer(self, image_array: np.ndarray = None, image_path: str = None) -> np.ndarray:
        """
        Run depth estimation on either a NumPy RGB array or an image path.

        Args:
            image_array (np.ndarray, optional): Input image as RGB float32 or uint8 array (HxWx3).
            image_path (str, optional): Path to an image file.
                One of these must be provided.

        Returns:
            np.ndarray: Float32 depth map (HxW).
        """
        if image_array is None and image_path is None:
            raise ValueError("You must provide either `image_array` or `image_path`.")

        # --- Convert NumPy input to PIL Image ---
        if image_array is not None:
            if image_array.dtype != np.uint8:
                # assume normalized [0,1] floats → scale back to [0,255]
                image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_array)
        else:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)

        # --- Preprocess & predict ---
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth[0].cpu().numpy().astype(np.float32)

        return depth

    # ------------------------------------------------------------------
    # Private: batch test helper (for internal validation only)
    # ------------------------------------------------------------------
    def _process_dir(self, image_dir: str, output_dir: str):
        """Internal batch helper for quick testing or dataset generation."""
        os.makedirs(output_dir, exist_ok=True)
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]

        if not image_files:
            print(f"[Error] No valid image files found in {image_dir}")
            return

        print(f"[Run] Found {len(image_files)} images in {image_dir}")

        for i, filename in enumerate(sorted(image_files), 1):
            path = os.path.join(image_dir, filename)
            name, _ = os.path.splitext(filename)
            print(f"[{i}/{len(image_files)}] Processing {filename}...")

            depth = self.infer(image_path=path)

            np.save(os.path.join(output_dir, f"{name}_raw.npy"), depth)

            depth_gray = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_MAGMA)

            cv2.imwrite(os.path.join(output_dir, f"{name}_gray.png"), depth_gray)
            cv2.imwrite(os.path.join(output_dir, f"{name}_color.png"), depth_color)

            print(f"[Save] Outputs saved for {filename}")

        print(f"[Done] Processed {len(image_files)} total images.")

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------
    @staticmethod
    def inspect_depth_file(npy_path: str):
        """Inspect stored depth map statistics."""
        depth = np.load(npy_path)
        print(f"Shape: {depth.shape}")
        print(f"Dtype: {depth.dtype}")
        print(f"Min: {depth.min():.2f}, Max: {depth.max():.2f}, Mean: {depth.mean():.2f}")
        print("Example values:", depth[::50, ::50])


# ----------------------------------------------------------------------
# Internal test run (manual)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    estimator = DepthEstimator()

    # Internal testing
    estimator._process_dir("./images/positives", "./outputs/positives")
    estimator._process_dir("./images/negatives", "./outputs/negatives")

    # Example single inference from array:
    # img = np.asarray(Image.open("./images/example.jpg").convert("RGB"))
    # depth = estimator.infer(image_array=img)
    # np.save("./outputs/example_raw.npy", depth)

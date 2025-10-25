"""
perception_pipeline.py
----------------------
End-to-end perception pipeline:
  - Loads both pretrained models (Depth + Segmentation)
  - Normalizes the image once
  - Runs inference through both models
  - Generates depth mask and computes IoU overlap
"""

import os
import cv2
import torch
import numpy as np
from typing import Optional, Dict, Any

# Local imports
from cv.depth_estimator import DepthEstimator
from cv.image_segmenter import PersonSegmenter
from cv.depth_mask_strategies import DepthMaskGenerator
from cv.utils import mask_overlap_percentage, dominant_quadrant


class PerceptionPipeline:
    """
    Unified depth-segmentation-overlap pipeline.
    """

    def __init__(
        self,
        model_depth_dir: str = "./cv/depth_model",
        masking_strategy: str = "fixed",
        score_thresh: float = 0.5,
        invert_seg: bool = False,
        target_size: tuple = (256, 256),
        near_thresh: float = 0.7,
        spray_thresh: float = 50.0,
        use_cuda: bool = True,
        return_keys: Optional[list[str]] = None,
    ):
        """
        Initialize all models and configuration.
        """
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        print(f"[Init] PerceptionPipeline using device: {self.device}")

        # Load pretrained models
        self.depth_model = DepthEstimator(model_dir=model_depth_dir, device=self.device)
        self.seg_model = PersonSegmenter(
            score_thresh=score_thresh,
            invert=invert_seg,
            target_size=target_size,
            device=self.device,
        )
        self.spray_thresh = spray_thresh
        self.target_size = target_size

        self.mask_generator = DepthMaskGenerator(method=masking_strategy, near_thresh=near_thresh)
        self.near_thresh = near_thresh
        self.return_keys = return_keys or ["overlap", "quadrant", "will_spray"]

    # Core normalization
    @staticmethod
    def _normalize_input(image_array: np.ndarray) -> np.ndarray:
        """
        Normalize input image to [0,1] float32 and ensure RGB format.
        """
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        elif image_array.dtype == np.float32 and image_array.max() > 1.0:
            image_array = np.clip(image_array / 255.0, 0, 1.0)
        elif image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)

        # If BGR (e.g., from OpenCV), convert to RGB
        if image_array.shape[-1] == 3 and np.mean(image_array[..., 0]) > np.mean(image_array[..., 2]):
            image_array = image_array[..., ::-1]  # swap channels
        return image_array

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------
    def infer(
        self,
        image_path: Optional[str] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:

        if image_array is None and image_path is None:
            raise ValueError("Must provide either image_array or image_path.")
        if image_array is None:
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise FileNotFoundError(f"Could not read image at {image_path}")

        # Normalize
        img_norm = self._normalize_input(image_array)

        # Run Model
        depth_map = self.depth_model.infer(image_array=img_norm)
        mask_seg = self.seg_model.infer(image_array=img_norm)

        quadrant = dominant_quadrant(mask_seg)

        # Use Generator for mask
        mask_depth = self.mask_generator.generate(depth_map)
        mask_depth = cv2.resize(mask_depth, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Intersection over union
        mask_depth_bin = (mask_depth > 127).astype(np.uint8)
        mask_seg_bin = (mask_seg > 127).astype(np.uint8)
        overlap = mask_overlap_percentage(mask_depth_bin, mask_seg_bin)

        # Build dictionary selectively
        outputs = {
            "overlap": overlap,
            "mask_depth": mask_depth,
            "mask_seg": mask_seg,
            "depth_map": depth_map,
            "normalized_image": img_norm,
            "quadrant": quadrant,
            "will_spray": overlap >= self.spray_thresh
        }

        # filter by self.return_keys
        return {k: v for k, v in outputs.items() if k in self.return_keys}

    # Optional visualization
    @staticmethod
    def visualize(
        mask_seg: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        mask_depth: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
    ):
        """
        Display visualization for debugging.

        Args:
            mask_seg (np.ndarray): Required binary segmentation mask (0/255).
            depth_map (np.ndarray, optional): Optional depth map to visualize.
            mask_depth (np.ndarray, optional): Optional binary depth mask.
            image_path (str, optional): Optional path to the original RGB image.
        """
        panels = []

        # Original RGB image
        if image_path is not None:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
            panels.append(img)

        # Depth colormap
        if depth_map is not None:
            depth_vis = cv2.applyColorMap(
                cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_MAGMA,
            )
            panels.append(depth_vis)

        # Depth mask
        if mask_depth is not None:
            panels.append(cv2.cvtColor(mask_depth, cv2.COLOR_GRAY2BGR))

        # Segmentation mask (required)
        panels.append(cv2.cvtColor(mask_seg, cv2.COLOR_GRAY2BGR))

        # Ensure consistent dimensions
        h, w = panels[0].shape[:2]
        panels = [cv2.resize(p, (w, h)) for p in panels]

        # Stack and display
        stacked = np.hstack(panels)
        window_title = " | ".join(
            lbl for lbl, p in zip(
                ["Original", "Depth", "Depth Mask", "Segmentation"], panels
            ) if p is not None
        )
        cv2.imshow(window_title, stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# Example usage
if __name__ == "__main__":
    pipeline = PerceptionPipeline(
        near_thresh=0.7,
        spray_thresh=50.0,
        return_keys=["overlap", "quadrant", "will_spray", "mask_seg", "mask_depth", "depth_map"],
    )

    # Loop over images 1–20
    for i in range(1, 10):
        image_path = f"./images/positives/{i}.png"
        if not os.path.exists(image_path):
            print(f"[Skip] Missing file: {image_path}")
            continue

        print(f"\n[Processing] {image_path}")

        result = pipeline.infer(image_path=image_path)

        print(f"[Result] Mask overlap = {result['overlap']:.2f}%")
        print(f"[Result] Dominant quadrant = {result['quadrant']}")
        print(f"[Result] Will spray = {result['will_spray']}")
        print('DONE!')

    # pipeline.visualize(
    #     result["depth_map"],
    #     result["mask_depth"],
    #     result["mask_seg"],
    # )
 

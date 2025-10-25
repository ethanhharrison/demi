"""
person_segmenter.py
-------------------
Encapsulates DeepLabV3-MobileNetV3 person segmentation in a class interface.
Supports both image-path and NumPy-array inference.

Default model: DeepLabV3-MobileNetV3-Large (COCO pretrained)
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


class PersonSegmenter:
    def __init__(
        self,
        score_thresh: float = 0.5,
        invert: bool = False,
        target_size=(256, 256),
        device: str = None,
    ):
        """
        Initialize the segmentation model and preprocessing transforms.

        Args:
            score_thresh (float): Confidence threshold for 'person' class.
            invert (bool): Whether to invert mask colors (white=person).
            target_size (tuple[int, int]): Resize output mask (H, W).
            device (str): Optional device override ("cuda" or "cpu").
        """
        self.score_thresh = score_thresh
        self.invert = invert
        self.target_size = target_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"[Init] Loading DeepLabV3-MobileNetV3 model on {self.device} ...")
        self.weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=self.weights).to(self.device).eval()
        self.transform = self.weights.transforms()
        print("[Ready] PersonSegmenter initialized.")

    # ------------------------------------------------------------------
    # Public: single image or array inference
    # ------------------------------------------------------------------
    def infer(self, image_array: np.ndarray = None, image_path: str = None) -> np.ndarray:
        """
        Run person segmentation on either a NumPy RGB array or an image path.

        Args:
            image_array (np.ndarray, optional): RGB image array (H×W×3, uint8 or float [0,1]).
            image_path (str, optional): Path to an image file.

        Returns:
            np.ndarray: Binary uint8 mask (H×W), 255=person, 0=background.
        """
        if image_array is None and image_path is None:
            raise ValueError("You must provide either `image_array` or `image_path`.")

        # --- Load or use provided image ---
        if image_array is not None:
            # handle float32 arrays normalized to [0,1]
            if image_array.dtype != np.uint8:
                image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
            # ensure RGB order (user responsibility to avoid BGR inputs)
            img = Image.fromarray(image_array)
        else:
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)

        # --- Transform + inference ---
        inp = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.model(inp)["out"].softmax(1)[0]  # [C,H,W]

        person_mask = out[15].cpu().numpy()  # COCO class 15 = person
        binary_mask = (person_mask > self.score_thresh).astype(np.uint8) * 255
        binary_mask = cv2.resize(binary_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.invert:
            binary_mask = 255 - binary_mask

        return binary_mask

    # ------------------------------------------------------------------
    # Private: batch folder helper (for testing only)
    # ------------------------------------------------------------------
    def _process_dir(self, input_dir: str, output_dir: str):
        """Internal batch helper for quick testing or dataset generation."""
        os.makedirs(output_dir, exist_ok=True)
        supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in supported_ext]

        if not images:
            print(f"No supported images found in {input_dir}")
            return

        print(f"Processing {len(images)} images from {input_dir} ...")
        for i, fname in enumerate(images, 1):
            in_path = os.path.join(input_dir, fname)
            out_name = os.path.splitext(fname)[0] + "_mask.png"
            out_path = os.path.join(output_dir, out_name)
            try:
                mask = self.infer(image_path=in_path)
                cv2.imwrite(out_path, mask)
                print(f"[{i}/{len(images)}] Saved: {out_path}")
            except Exception as e:
                print(f"[{i}/{len(images)}] Error on {fname}: {e}")

        print("\nAll masks saved to:", os.path.abspath(output_dir))


# ----------------------------------------------------------------------
# Internal test run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    segmenter = PersonSegmenter(score_thresh=0.5, invert=False, target_size=(256, 256))

    # Internal folder test (optional)
    segmenter._process_dir("./images/positives", "./segmentation_images/positives")
    segmenter._process_dir("./images/negatives", "./segmentation_images/negatives")

    # Example: run directly on loaded NumPy array
    # import cv2
    # frame = cv2.imread("./images/example.jpg")[:, :, ::-1]  # BGR→RGB
    # mask = segmenter.infer(image_array=frame)
    # cv2.imwrite("./segmentation_images/example_mask.png", mask)

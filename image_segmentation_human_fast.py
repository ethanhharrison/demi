import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

# ---------------------------------------------------------------------
# Load pretrained model (once)
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights).to(device).eval()
transform = weights.transforms()

# ---------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------
def get_binary_person_mask(image_path: str, score_thresh: float = 0.5) -> np.ndarray:
    """Generate a binary (black/white) mask where 255 = person, 0 = background."""
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    inp = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(inp)["out"].softmax(1)[0]  # shape [C, H, W]

    person_mask = out[15].cpu().numpy()  # COCO class 15 = person
    binary_mask = (person_mask > score_thresh).astype(np.uint8) * 255
    return binary_mask

# ---------------------------------------------------------------------
# Folder processing
# ---------------------------------------------------------------------
def process_folder(input_dir: str, output_dir: str, score_thresh: float = 0.5):
    """Process all images in a folder and save binary person masks."""
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
            mask = get_binary_person_mask(in_path, score_thresh)
            cv2.imwrite(out_path, mask)
            print(f"[{i}/{len(images)}] Saved: {out_path}")
        except Exception as e:
            print(f"[{i}/{len(images)}] Error on {fname}: {e}")

    print("\nAll masks saved to:", os.path.abspath(output_dir))

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    input_dir = "./images/positives"                     # folder of RGB images
    output_dir = "./segmentation_images/positives"       # where masks will be saved
    process_folder(input_dir, output_dir, score_thresh=0.5)
    input_dir = "./images/negatives"                     # folder of RGB images
    output_dir = "./segmentation_images/negatives"       # where masks will be saved
    process_folder(input_dir, output_dir, score_thresh=0.5)
# ≈ 11 million ~43 MB
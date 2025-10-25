"""
depth_perception_batch.py
-------------------------
Downloads Intel's DPT depth-estimation model (if missing),
runs depth inference on every image in ./images/,
and saves outputs (raw, grayscale, colorized) for each.
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor




# ---------------------------------------------------------------------
# 1. Download or load the model
# ---------------------------------------------------------------------
def download_or_load():
    MODEL_ID = "Intel/dpt-swinv2-tiny-256"
    MODEL_DIR = "./depth_model"

    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"[Load] Found local model at {MODEL_DIR}")
        return AutoModelForDepthEstimation.from_pretrained(MODEL_DIR)

    print(f"[Setup] Downloading {MODEL_ID} model weights...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.save_pretrained(MODEL_DIR)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    processor.save_pretrained(MODEL_DIR)

    print(f"[Done] Model + processor saved at {MODEL_DIR}")
    return model


# ---------------------------------------------------------------------
# 2. Run inference on a single image
# ---------------------------------------------------------------------
from PIL import Image, ImageOps

def run_depth_inference(image_path: str, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)


    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth[0].cpu().numpy().astype(np.float32)

    return depth


# ---------------------------------------------------------------------
# 3. Batch process all images
# ---------------------------------------------------------------------
def run_batch_inference(image_dir, output_dir):
    IMAGE_DIR = image_dir
    OUTPUT_DIR = output_dir
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure folders exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model + processor once
    model = download_or_load().to(DEVICE)
    processor = AutoImageProcessor.from_pretrained("./depth_model", use_fast=True)

    # Gather all image files
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_ext)]

    if not image_files:
        print(f"[Error] No valid image files found in {IMAGE_DIR}")
        return

    print(f"[Run] Found {len(image_files)} images in {IMAGE_DIR}")

    for filename in image_files:
        image_path = os.path.join(IMAGE_DIR, filename)
        name, _ = os.path.splitext(filename)
        print(f"[Process] {filename}")

        # Run inference
        depth = run_depth_inference(image_path, model, processor, DEVICE)

        # Save results
        np.save(os.path.join(OUTPUT_DIR, f"{name}_raw.npy"), depth)

        depth_gray = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_MAGMA)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_gray.png"), depth_gray)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_color.png"), depth_color)

        print(f"[Save] Outputs for {filename} saved in {OUTPUT_DIR}")

    print(f"[Done] Processed {len(image_files)} images total.")

def inspect_depth_files(npy_files_for_masking_depth):
    import numpy as np

    depth = np.load(npy_files_for_masking_depth)
    print("Shape:", depth.shape)
    print("Dtype:", depth.dtype)
    print("Min:", depth.min())
    print("Max:", depth.max())
    print("Mean:", depth.mean())
    print("Example values:", depth[::50, ::50])  # sample every 50 px


# Metric	Value	Interpretation
# Shape	(256, 256)	The model resized your input image to 256×256 pixels before inference. Each pixel corresponds to a depth estimate.
# Dtype	float32	Standard 32-bit floating-point depth values.
# Min	≈ 35.27	Closest predicted region.
# Max	≈ 2358.67	Farthest predicted region.
# Mean	≈ 1094.28	The average depth value across the entire image.
# Example values	30 – 2300 range	Shows both shallow (tens) and deep (thousands) pixels in the same frame — high contrast in depth.



def masking_depth_perception(npy_files_for_masking_depth):
    pass
# ---------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # inspect_depth_files('/Users/ericji/Desktop/Repos/demi/outputs/Screenshot 2025-10-24 at 9.07.59 PM_raw.npy')
    run_batch_inference(image_dir="./images/positives", output_dir="./outputs/positives")
    run_batch_inference(image_dir="./images/negatives", output_dir="./outputs/negatives")


# 164 mb

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# Load model once (COCO pretrained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
transform = weights.transforms()

def get_person_mask(image_path: str, thr: float = 0.5, min_area: int = 128) -> np.ndarray:
    """Run Mask R-CNN on a single image and return a combined binary mask for all people."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    x = transform(img).to(device)

    with torch.no_grad():
        pred = model([x])[0]

    mask_total = np.zeros((H, W), dtype=np.uint8)
    for label, score, mask in zip(pred["labels"], pred["scores"], pred["masks"]):
        if label.item() != 1 or score.item() < thr:
            continue
        m = F.interpolate(mask[None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        m = (m.cpu().numpy() >= 0.5).astype(np.uint8)
        if m.sum() < min_area:
            continue
        mask_total = np.maximum(mask_total, m)
    return mask_total

def save_person_mask(image_path: str, out_path: str = None) -> str:
    """Generate and save a binary person mask (0/255). Returns output path."""
    mask = get_person_mask(image_path)
    if out_path is None:
        name, _ = os.path.splitext(image_path)
        out_path = f"{name}_mask.png"
    cv2.imwrite(out_path, mask * 255)
    print(f"Saved mask to {out_path}")
    return out_path

if __name__ == "__main__":
    import sys
    depth = 'first'
    image_path = f"./images/{depth}.jpeg"
    save_person_mask(image_path)

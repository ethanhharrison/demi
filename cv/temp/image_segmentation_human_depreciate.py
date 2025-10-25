import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# ---------------------------------------------------------------------
# Load model once (global)
# ---------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
_model = maskrcnn_resnet50_fpn(weights=_weights).to(_device).eval()
_transform = _weights.transforms()

# ---------------------------------------------------------------------
# Fast binary person mask extractor
# ---------------------------------------------------------------------
def get_binary_person_mask(image_path: str, score_thresh: float = 0.5) -> np.ndarray:
    """
    Fastest possible path to get a binary (black/white) person mask from an image.

    Args:
        image_path (str): Path to input RGB image.
        score_thresh (float): Minimum confidence for detections.

    Returns:
        np.ndarray: Binary mask (uint8, shape HxW, values {0,255})
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x = _transform(img).to(_device)

    with torch.inference_mode():
        pred = _model([x])[0]

    # Select only "person" masks (label == 1) above threshold
    keep = (pred["labels"] == 1) & (pred["scores"] >= score_thresh)
    if not keep.any():
        return np.zeros((h, w), dtype=np.uint8)

    masks = pred["masks"][keep]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    masks = (masks.squeeze(1).cpu().numpy() >= 0.5).astype(np.uint8)

    # Combine all person masks into one binary array
    combined = np.clip(masks.sum(axis=0), 0, 1).astype(np.uint8)

    return combined * 255  # 0=black, 255=white

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mask = get_binary_person_mask("./images/first.jpeg")
    cv2.imwrite("person_mask_fast.png", mask)
    print("Saved binary mask -> person_mask_fast.png")

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


# ---------------------------------------------------------------------
# Load pretrained model once
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
transform = weights.transforms()


# ---------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------
def people(img, thr=0.5, min_area=128):
    """Return list of detected person instances with mask, score, and box."""
    W, H = img.size
    x = transform(img).to(device)
    with torch.no_grad():
        preds = model([x])[0]

    out = []
    for label, score, mask, box in zip(
        preds["labels"], preds["scores"], preds["masks"], preds["boxes"]
    ):
        if label.item() != 1 or score.item() < thr:
            continue
        m = F.interpolate(mask[None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        m = (m.cpu().numpy() >= 0.5).astype(np.uint8)
        if m.sum() < min_area:
            continue
        out.append({"mask": m, "score": float(score), "box": tuple(map(float, box))})
    return out


def overlay(img, inst, alpha=0.55):
    """Overlay detected masks on the original RGB image."""
    palette = np.array([
        [220, 20, 60], [65, 105, 225], [60, 179, 113],
        [255, 165, 0], [138, 43, 226], [70, 130, 180],
        [255, 99, 71], [154, 205, 50]
    ], dtype=np.float32)

    x = np.array(img.convert("RGB"), dtype=np.float32)
    for i, d in enumerate(inst):
        m = d["mask"].astype(bool)
        x[m] = (1 - alpha) * x[m] + alpha * palette[i % len(palette)]
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def save_persons(img, inst, out_dir="cuts"):
    """Save each detected person as an RGBA PNG crop."""
    os.makedirs(out_dir, exist_ok=True)
    for i, d in enumerate(inst, 1):
        m = (d["mask"] * 255).astype(np.uint8)
        rgba = Image.new("RGBA", img.size)
        rgba.paste(img, mask=Image.fromarray(m))
        out_path = os.path.join(out_dir, f"person_{i:02d}.png")
        rgba.save(out_path)


def save_faces(img, inst, out_dir="faces", scale=1.1):
    """Save face crops for each detected person."""
    os.makedirs(out_dir, exist_ok=True)
    a = np.array(img.convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(g, scaleFactor=1.15, minNeighbors=5)
    if len(faces) == 0:
        return 0

    H, W = g.shape
    boxes = []
    for d in inst:
        ys, xs = np.where(d["mask"] > 0)
        if xs.size == 0:
            boxes.append((0, 0, 0, 0))
        else:
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    counts = [0] * len(inst)
    n = 0
    for (x, y, w, h) in faces:
        cx, cy = x + w / 2, y + h / 2
        rw, rh = w * scale, h * scale
        x1, y1 = max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2))
        x2, y2 = min(W, int(cx + rw / 2)), min(H, int(cy + rh / 2))

        best, best_iou = -1, 0.0
        for i, (ax1, ay1, ax2, ay2) in enumerate(boxes):
            ix1, iy1 = max(x1, ax1), max(y1, ay1)
            ix2, iy2 = min(x2, ax2), min(y2, ay2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (x2 - x1) * (y2 - y1)
            a2 = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > best_iou:
                best_iou, best = iou, i
        if best == -1:
            continue

        pm = inst[best]["mask"]
        sub = (pm[y1:y2, x1:x2] * 255).astype(np.uint8)
        if sub.sum() == 0:
            continue

        crop = img.crop((x1, y1, x2, y2))
        rgba = Image.new("RGBA", crop.size)
        rgba.paste(crop, mask=Image.fromarray(sub))
        counts[best] += 1
        n += 1
        out_path = os.path.join(out_dir, f"person_{best+1:02d}_face_{counts[best]:02d}.png")
        rgba.save(out_path)
    return n


# ---------------------------------------------------------------------
# Unified run function
# ---------------------------------------------------------------------
def run_segmentation(image_path: str):
    """Given one image path, detect people, save overlay, cropped persons, and faces."""
    img = Image.open(image_path).convert("RGB")
    name = os.path.splitext(os.path.basename(image_path))[0]

    instances = people(img, thr=0.55)
    if not instances:
        print(f"No persons detected in {image_path}")
        return

    overlay_img = overlay(img, instances)
    overlay_img.save(f"overlay_{name}.png")
    print(f"Saved overlay_{name}.png")

    save_persons(img, instances, f"cuts_{name}")
    print(f"Saved person crops to cuts_{name}/")

    faces_found = save_faces(img, instances, f"faces_{name}")
    print(f"Saved faces ({faces_found}) to faces_{name}/")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python3 run_segmentation_one_image.py <image_path>")
    #     sys.exit(1)
    run_segmentation("./images/first.jpeg")

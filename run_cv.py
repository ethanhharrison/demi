"""
run_cv.py
---------
Run DEMI perception pipeline on one image.
"""

from cv.cv_pipeline import PerceptionPipeline

# Choose a single image
IMAGE_PATH = "./images/positives/2.png"
IMAGE_PATH2 = "./images/negatives/3.png"
VISUALIZE = True
# Initialize the perception pipeline
pipeline = PerceptionPipeline(
    near_thresh=0.7,
    spray_thresh=50.0,
    return_keys=["overlap", "quadrant", "will_spray", "mask_seg", "mask_depth", "depth_map"],
)

# Run inference on one image
result = pipeline.infer(image_path=IMAGE_PATH)

# Print results
print(f"[Result] Mask overlap = {result['overlap']:.2f}%")
print(f"[Result] Dominant quadrant = {result['quadrant']}")
print(f"[Result] Will spray = {result['will_spray']}")



# ------------- RUN SECOND IMAGE ----------

# Run inference on one image
result = pipeline.infer(image_path=IMAGE_PATH2)

# Print results
print(f"[Result] Mask overlap = {result['overlap']:.2f}%")
print(f"[Result] Dominant quadrant = {result['quadrant']}")
print(f"[Result] Will spray = {result['will_spray']}")

# Optional visualization
if VISUALIZE:
    pipeline.visualize(
        depth_map=result["depth_map"],
        mask_depth=result["mask_depth"],
        mask_seg=result["mask_seg"],
    )

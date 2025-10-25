"""
run_offensive_cv.py
---------
Run DEMI perception pipeline on one image.
"""

from cv.cv_pipeline import PerceptionPipeline

def predict_offensive_spray(
        near_thresh: float = 0.7, 
        spray_thresh: float = 50.0, 
        image_path = None, 
        image_array = None
    ):
    pipeline = PerceptionPipeline(
        near_thresh=near_thresh,
        spray_thresh=spray_thresh,
        return_keys=["overlap", "quadrant", "will_spray", "mask_seg", "mask_depth", "depth_map"],
    )
    # Run inference on one image
    if image_path:
        result = pipeline.infer(image_path=image_path)
    elif image_array:
        result = pipeline.infer(image_path=image_array)
    else:
        raise Exception("Must provide either an image path or image array from prediction.")

    return result["quadrant"], result["will_spray"]

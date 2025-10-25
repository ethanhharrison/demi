"""
Camera capture utilities for capturing frames from a webcam.

This module provides functions to initialize a webcam, capture frames,
and save them to disk with timestamps.
"""

import cv2
import os
from datetime import datetime
from typing import Optional, Tuple


def initialize_camera(camera_id: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Initialize and open a webcam capture device.
    
    Args:
        camera_id: The camera device ID (default: 0 for primary webcam).
    
    Returns:
        VideoCapture object if successful, None if the camera could not be opened.
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        return None
    
    return cap


def ensure_save_directory(save_dir: str = "captures") -> str:
    """
    Create the directory for saving captured frames if it doesn't exist.
    
    Args:
        save_dir: Path to the directory where frames will be saved (default: "captures").
    
    Returns:
        The absolute path to the save directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    return os.path.abspath(save_dir)


def capture_frame(cap: cv2.VideoCapture) -> Optional[Tuple[bool, any]]:
    """
    Capture a single frame from the video capture device.
    
    Args:
        cap: An active VideoCapture object.
    
    Returns:
        Tuple of (success, frame) where success is True if frame was captured,
        None if capture object is invalid.
    """
    if cap is None or not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    return (ret, frame)


def save_frame(frame: any, save_dir: str = "captures", prefix: str = "frame") -> Optional[str]:
    """
    Save a captured frame to disk with a timestamp-based filename.
    
    Args:
        frame: The image frame to save (numpy array from OpenCV).
        save_dir: Directory where the frame will be saved (default: "captures").
        prefix: Filename prefix (default: "frame").
    
    Returns:
        The full path to the saved file, or None if save failed.
    """
    if frame is None:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(save_dir, f"{prefix}_{timestamp}.jpg")
    
    success = cv2.imwrite(filename, frame)
    
    if success:
        return filename
    return None


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    """
    Release the video capture device and free resources.
    
    Args:
        cap: The VideoCapture object to release (can be None).
    """
    if cap is not None:
        cap.release()


def capture_and_save_frame(cap: cv2.VideoCapture, save_dir: str = "captures") -> Optional[str]:
    """
    Convenience function to capture a frame and immediately save it.
    
    Args:
        cap: An active VideoCapture object.
        save_dir: Directory where the frame will be saved (default: "captures").
    
    Returns:
        The path to the saved file if successful, None otherwise.
    """
    result = capture_frame(cap)
    
    if result is None:
        return None
    
    ret, frame = result
    
    if not ret:
        return None
    
    return save_frame(frame, save_dir)
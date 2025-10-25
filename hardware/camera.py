import cv2
import os
from datetime import datetime

# === CONFIGURATION ===
save_dir = "webcam_captures"  # Change to your desired path
os.makedirs(save_dir, exist_ok=True)

# === CAMERA SETUP ===
cap = cv2.VideoCapture(0)  # Use 0 or change to 1 if needed

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press Ctrl+C to stop capturing.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        print(f"Saved {filename}")

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Stopping capture...")

finally:
    cap.release()
    print("Webcam released. All done!")


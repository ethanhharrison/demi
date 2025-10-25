import json
import time
import base64
import numpy as np
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

LOG_PATH = "/Users/eharrison/data/sensor_log.json"

class LogHandler(FileSystemEventHandler):
    def __init__(self, response_fn):
        self.last_size = 0
        self.response_fn = response_fn

    def on_modified(self, event):
        if event.src_path != LOG_PATH:
            return
        try:
            with open(LOG_PATH, "r") as f:
                f.seek(self.last_size)
                new_data = f.read()
                self.last_size = f.tell()
        except FileNotFoundError:
            return

        line = new_data.strip().splitlines()[-1]
        try:
            entry = json.loads(line)
            self.response_fn(entry)
        except Exception as e:
            print(f"(Malformed line ignored) {e}")

def b64_to_image(frame_b64):
    decoded_bytes = base64.b64decode(frame_b64) 
    decoded_arr = np.frombuffer(decoded_bytes, np.uint8)
    decoded_img = cv2.imdecode(decoded_arr, cv2.IMREAD_COLOR)
    return decoded_img

if __name__ == "__main__":
    response = lambda e: print(f"{e['datetime']} | Smelly: {e['smelly']} | frame_b64: {e.get('frame_b64')}")
    handler = LogHandler(response)
    observer = Observer()
    observer.schedule(handler, path="/Users/eharrison/data", recursive=False)
    observer.start()
    print(f"Watching {LOG_PATH} for updates...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
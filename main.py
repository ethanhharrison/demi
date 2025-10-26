import time
from watchdog.observers import Observer

from control.run_trajectory import take_trajectory
from cv.run_offensive_cv import predict_offensive_spray
from server.collect_pi_data import LogHandler, b64_to_image

LOG_PATH = "/Users/eharrison/data"

def handle_sensor_data(entry):
    _, smelly, frame_b64 = entry["datetime"], entry["smelly"], entry.get("frame_b64")
    if smelly:
        # Run self-spray trajectory
        pass
    elif frame_b64:
        image = b64_to_image(frame_b64)
        quadrant, will_spray = predict_offensive_spray(image)
        if will_spray:
            # Run offensive-spray trajectory based on quadrant
            pass
    else:
        print("No Spray")

def main():
    handler = LogHandler(handle_sensor_data)
    observer = Observer()
    observer.schedule(handler, path=LOG_PATH, recursive=False)
    observer.start()
    print(f"Watching {LOG_PATH} for updates...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
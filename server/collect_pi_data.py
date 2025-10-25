import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

LOG_PATH = "/Users/eharrison/data/sensor_log.json"

class LogHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_size = 0

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

        # Split and print each new JSON entry
        for line in new_data.strip().splitlines():
            try:
                entry = json.loads(line)
                print(f"{entry['datetime']} | Smelly: {entry['smelly']}")
            except Exception as e:
                print(f"(Malformed line ignored) {e}")

if __name__ == "__main__":
    handler = LogHandler()
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
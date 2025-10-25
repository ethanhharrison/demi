from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time, pandas as pd

class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".csv"):
            print("New data received!")
            df = pd.read_csv(event.src_path)
            print(df.head())

observer = Observer()
observer.schedule(Handler(), "/Users/alex/data", recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
import time
import json
import os
import paramiko
from watchdog.observers import Observer

from control.run_trajectory import take_trajectory
from cv.run_offensive_cv import predict_offensive_spray
from server.collect_pi_data import LogHandler, b64_to_image

LOG_PATH = "/Users/eharrison/data"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".config.json")

def load_config():
    """Safely load hidden Raspberry Pi SSH config."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {CONFIG_PATH}! Please create .config.json.")
    except json.JSONDecodeError:
        raise ValueError("Malformed .config.json")
    
config = load_config

def send_flag_over_ssh(flag: str):
    """Send a control flag to Raspberry Pi using config file credentials."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(
            config["PI_IP"],
            username=config["PI_USER"],
        )

        sftp = ssh.open_sftp()
        with sftp.file(config["PI_FLAG_PATH"], "w") as f:
            f.write(flag + "\n")
        sftp.close()
        ssh.close()

        print(f"[SSH] Sent flag '{flag}' to Pi at {config['PI_IP']}")
    except Exception as e:
        print(f"[ERROR] SSH send failed: {e}")

def handle_sensor_data(entry):
    _, smelly, frame_b64 = entry["datetime"], entry["smelly"], entry.get("frame_b64")
    if smelly:
        send_flag_over_ssh("IN PROGRESS")
        # TODO: Run self-spray trajectory
        try:
            print("Taking Trajectory...")
        except Exception as e:
            print(f"Error taking trajectory: {e}")
        send_flag_over_ssh("COMPLETE")
        return
    elif frame_b64:
        image = b64_to_image(frame_b64)
        quadrant, will_spray = predict_offensive_spray(image)
        if will_spray:
            # TODO: Run offensive-spray trajectory based on quadrant
            try:
                print(f"Taking Offensive Spray Trajectory in Quadrant {quadrant}...")
            except Exception as e:
                print(f"Error taking offensive trajectory: {e}")
            return
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
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
    
config = load_config()

def send_flag_over_ssh(flag: str):
    """Send a control flag to Raspberry Pi using config file credentials."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(config["PI_IP"], username=config["PI_USER"])

    sftp = ssh.open_sftp()
    with sftp.file(config["PI_FLAG_PATH"], "w") as f:
        f.write(flag + "\n")
    sftp.close()
    ssh.close()

    print(f"[SSH] Sent flag '{flag}' to Pi at {config['PI_IP']}")


TRAJECTORY_MAP = {
    "upper_left": 0,
    "upper_right": 1,
    "lower_left": 2,
    "lower_right": 3,
}

def do_spray_action(smelly, frame_b64):
    if smelly:
        take_trajectory(TRAJECTORY_MAP["lower_left"])
        time.sleep(1)
        take_trajectory(TRAJECTORY_MAP["lower_right"])
        time.sleep(1)
    elif frame_b64:
        image = b64_to_image(frame_b64)
        quadrant, will_spray = predict_offensive_spray(image_array=image)
        print(quadrant, will_spray)
        if will_spray and quadrant in [1, 2, 4, 5, 7, 8]: 
            take_trajectory(TRAJECTORY_MAP["upper_left"])
            time.sleep(1)
        if will_spray and quadrant in [2, 3 ,5, 6, 8, 9]:
            take_trajectory(TRAJECTORY_MAP["upper_right"])
            time.sleep(1)
    else:
        print("No Spray")
    
def handle_sensor_data(entry):
    timestamp, smelly, frame_b64 = entry["datetime"], entry["smelly"], entry.get("frame_b64")
    send_flag_over_ssh("IN PROGRESS")
    do_spray_action(smelly, frame_b64)
    send_flag_over_ssh("COMPLETE")

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
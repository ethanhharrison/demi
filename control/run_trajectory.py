import subprocess
import time

def take_trajectory(episode_num: int = 0, repo_id: str = "demi_trajectories"):
    subprocess.run([
        "lerobot-replay",
        "--robot.type=so101_follower",
        "--robot.port=/dev/tty.usbmodem5A7A0186521",
        "--robot.id=my_white_robot_arm",
        f"--dataset.repo_id=ehharrison/{repo_id}",
        f"--dataset.episode={episode_num}"
    ])

if __name__ == "__main__":
    for i in range(8):
        take_trajectory(i)
        time.sleep(3)
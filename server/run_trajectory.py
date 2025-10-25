import subprocess

def take_trajectory(episode_num):
    subprocess.run([
        "lerobot-replay",
        "--robot.type=so101_follower",
        "--robot.port=/dev/tty.usbmodem5A7A0186521",
        "--robot.id=my_white_robot_arm",
        "--dataset.repo_id=ehharrison/demi_trajectories",
        f"--dataset.episode={episode_num}"
    ])

if __name__ == "__main__":
    take_trajectory(0)
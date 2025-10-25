import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

episode_idx = 0

robot_config = SO100FollowerConfig(port="/dev/tty.usbmodem5A7A0186521", id="my_white_robot_arm")

robot = SO100Follower(robot_config)
robot.connect()

dataset = LeRobotDataset("ehharrison/demi_trajectories", episodes=[episode_idx])
actions = dataset.hf_dataset.select_columns("action")

log_say(f"Replaying episode {episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

robot.disconnect()
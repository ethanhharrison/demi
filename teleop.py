from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5A7A0186521",
    id="my_white_robot_arm",
)

teleop_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5A4B0479551",
    id="my_black_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

print("Starting teleop. Use Ctrl+C to exit")
while True:
    action = teleop_device.get_action()
    robot.send_action(action)
#!/usr/bin/env bash
set -eo pipefail
cd /home/guest/pengyang
# ros2 launch resolves the disabled Wuji drivers/control from this colcon
# workspace. Pixi supplies Python dependencies but does not populate the ROS
# package index by itself, so preserve the sourced overlay for child launches.
source /opt/ros/foxy/setup.bash
source /home/guest/pengyang/install/setup.bash
set -u
exec /home/guest/.pixi/bin/pixi run --locked python replay_hand/replay_hand.py "$@"

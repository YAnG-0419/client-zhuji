#!/usr/bin/env bash
set -eo pipefail
cd /home/guest/pengyang
source /opt/ros/foxy/setup.bash
source /home/guest/pengyang/install/setup.bash
set -u
exec /home/guest/.pixi/bin/pixi run --locked python replay_hand/replay_hand_pinocchio.py "$@"

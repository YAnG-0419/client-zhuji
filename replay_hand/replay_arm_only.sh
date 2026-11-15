#!/usr/bin/env bash
set -euo pipefail
cd /home/guest/pengyang
# No ROS environment or Wuji driver startup is needed for arm-only replay.
exec /home/guest/.pixi/bin/pixi run --locked python replay_hand/replay_arm_only.py "$@"

#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/foxy/setup.bash
source "$ROOT/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}
export PYTHONNOUSERSITE=1
cd "$ROOT"
exec /home/guest/.pixi/bin/pixi run --locked python "$ROOT/replay/replay_servop.py" "$@"

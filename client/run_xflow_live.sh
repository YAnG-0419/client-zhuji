#!/usr/bin/env bash
# zhuji entry for the XFlow online client.  Default is a shadow run (no commands);
# add --execute to move the robot.  Requires the Mac relay (mac_relay.sh) to be up.
set -eo pipefail
cd /home/guest/pengyang
source /opt/ros/foxy/setup.bash
source /home/guest/pengyang/install/setup.bash
set -u
# Same DDS transport as the camera drivers in /opt/ros-bringup, otherwise the
# head image topic is not discovered.
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/ros-bringup/fastdds_config_udp_only.xml
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 MUJOCO_GL=disable
# Small matrices: multithreaded BLAS only adds contention with the 300 Hz sender.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/guest/.pixi/bin/pixi run --locked python client/live_client.py "$@"

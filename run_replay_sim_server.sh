#!/usr/bin/env bash
set -eo pipefail
cd /home/guest/pengyang
source /opt/ros/foxy/setup.bash
source /home/guest/pengyang/install/setup.bash
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PENGYANG_ROOT=/home/guest/pengyang
export PYTHONPATH=/home/guest/pengyang/client:/home/guest/pengyang/replay_hand:/home/guest/pengyang/replay:/home/guest/pengyang/tools:/home/guest/pengyang
exec /home/guest/.pixi/bin/pixi run --locked python client/replay_sim_server.py "$@"

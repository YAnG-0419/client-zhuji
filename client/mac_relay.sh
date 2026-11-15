#!/usr/bin/env bash
# Run on the Mac (it can reach both b300 and zhuji):  ./mac_relay.sh [5090-job-name]
# Reads the job's current endpoint.json and forwards zhuji 127.0.0.1:28765 to it.
set -euo pipefail
JOB=${1:-aw-hha12-tidy-5090-1114}
HOST=$(ssh b300 "python3 -c 'import json;print(json.load(open(\"/mnt/workspace/users/pengyang/workspace/outputs/$JOB/endpoint.json\"))[\"host\"])'")
echo "XFlow server $JOB -> $HOST:18765"
exec python3 "$(cd "$(dirname "$0")" && pwd)/relay_mac.py" --server-host "$HOST"

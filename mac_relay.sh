#!/usr/bin/env bash
# Run on the Mac (it can reach both b300 and zhuji).
#   ./mac_relay.sh <job>[/<service>]
# examples:
#   ./mac_relay.sh xflow-4srv-te-5090-1115/flower-hh      # multi-service pod
#   ./mac_relay.sh aw-hha12-tidy-5090-1114                # one service per job
# Reads that service's current endpoint.json (host AND port) and forwards
# zhuji 127.0.0.1:28765 to it.
set -euo pipefail
TARGET=${1:?usage: ./mac_relay.sh <job>[/<service>]}
ENDPOINT=/mnt/workspace/users/pengyang/workspace/outputs/$TARGET/endpoint.json
read -r HOST PORT MODELS < <(ssh b300 "python3 -c 'import json;d=json.load(open(\"$ENDPOINT\"));print(d[\"host\"],d[\"port\"],\",\".join(d.get(\"models\",[])))'")
echo "XFlow $TARGET -> $HOST:$PORT  models: $MODELS"
exec python3 "$(cd "$(dirname "$0")" && pwd)/relay_mac.py" --server-host "$HOST" --server-port "$PORT"

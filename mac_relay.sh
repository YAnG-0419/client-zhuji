#!/usr/bin/env bash
# Run on the Mac (the only host that reaches both b300 and zhuji).
#
#   ./mac_relay.sh <job>                 all checkpoints of that job, one port each
#   ./mac_relay.sh <job>/<service>       just that one service
#
# <job> is the OUTPUT DIRECTORY name under
# /mnt/workspace/users/pengyang/workspace/outputs (printed by the submit tool),
# which is not always the squeue job name.  Each service's endpoint.json gives
# the pod host and its server port; the local port mirrors the server port, so
# server 18765..18768 become zhuji 28765..28768 and the robot picks one with
#   bash client/run_xflow_live.sh --relay-port 2876N ...
set -euo pipefail
TARGET=${1:?usage: ./mac_relay.sh <job>[/<service>]}
ROOT=/mnt/workspace/users/pengyang/workspace/outputs
HERE="$(cd "$(dirname "$0")" && pwd)"

# One ssh round trip: list every endpoint under the target as "service host srv local".
ROUTES=()
while IFS= read -r line; do [ -n "$line" ] && ROUTES+=("$line"); done < <(ssh b300 "python3 - <<'PY'
import json, pathlib, sys
target = pathlib.Path('$ROOT/$TARGET')
if not target.is_dir():
    sys.exit('no such output directory: %s' % target)
files = sorted(target.glob('*/endpoint.json')) or (
    [target / 'endpoint.json'] if (target / 'endpoint.json').is_file() else [])
if not files:
    sys.exit('no endpoint.json under %s; is the server still loading?' % target)
rows = []
for path in files:
    d = json.loads(path.read_text())
    port = int(d['port'])
    name = path.parent.name if path.parent != target else target.name
    rows.append((port, name, d['host'], ','.join(d.get('models', []))))
for port, name, host, models in sorted(rows):
    print(name, host, port, 28765 + port - 18765, models)
PY")

if ((${#ROUTES[@]} == 0)); then echo "no services found for $TARGET" >&2; exit 1; fi
ARGS=()
printf '%-18s %-14s %-7s %-7s %s\n' SERVICE HOST SERVER LOCAL MODEL
for route in "${ROUTES[@]}"; do
  read -r name host server local models <<<"$route"
  printf '%-18s %-14s %-7s %-7s %s\n' "$name" "$host" "$server" "$local" "$models"
  ARGS+=(--route "$host:$server:$local")
done
exec python3 "$HERE/relay_mac.py" "${ARGS[@]}"

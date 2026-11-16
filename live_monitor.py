#!/usr/bin/env python3
"""Serve the run report live while live_client.py is still running.

Read-only: it only watches the audit directory that live_client's background
sink already writes, so it cannot slow down or disturb the control loop.  A new
chunk shows up about a chunk after it happened (the plan as soon as it is
planned, the tracking curves when the chunk finishes).

    pixi run --locked python client/live_monitor.py            # newest run, :8420
    pixi run --locked python client/live_monitor.py --run RUN --port 8420

From the Mac:  ssh -N -L 8420:127.0.0.1:8420 zhuji   then open
http://127.0.0.1:8420/
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time

from visualize_run import (build_cycle, load_execution, load_plan, page,
                           virtual_intrinsics)

POLL_S = 0.4
KEEPALIVE_S = 15.0

LIVE_SCRIPT = """
let follow = true;
const badge = document.createElement('span');
badge.style.cssText = 'margin-left:10px;padding:1px 6px;border-radius:3px;font-size:11px';
document.querySelector('h1').appendChild(badge);
const toggle = document.createElement('label');
toggle.style.cssText = 'margin-left:10px;font-size:11px;color:#8b97a8;cursor:pointer';
toggle.innerHTML = '<input type="checkbox" id="follow" checked> 跟随最新段';
document.querySelector('h1').appendChild(toggle);
document.getElementById('follow').onchange = e => { follow = e.target.checked; };
function setBadge(text, colour){
  badge.textContent = text;
  badge.style.background = colour; badge.style.color = '#0b0e13';
}
setBadge('连接中', '#8b97a8');

function apply(message){
  if (message.meta) { Object.assign(DATA, message.meta); renderMeta(); }
  if (message.cycle){
    const atEnd = cycle >= DATA.cycles.length - 1;
    const index = DATA.cycles.findIndex(c => c.plan_id === message.cycle.plan_id);
    if (index >= 0) DATA.cycles[index] = message.cycle; else DATA.cycles.push(message.cycle);
    buildTable();
    const target = (follow && atEnd) ? DATA.cycles.length - 1
                 : (index >= 0 ? cycle : Math.min(cycle, DATA.cycles.length - 1));
    if (target >= 0) select(target);
  }
  if (message.state) setBadge(message.state, message.state === '运行中' ? '#7fd88f' : '#ffd479');
}

function connect(){
  const feed = new EventSource('events');
  feed.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.reload) { location.reload(); return; }
    if (message.keepalive) return;
    apply(message);
  };
  feed.onerror = () => { setBadge('断开', '#ff8f8f'); feed.close(); setTimeout(connect, 2000); };
  feed.onopen = () => setBadge('已连接', '#7fd88f');
}
connect();
"""


def newest_run(runs: Path) -> Path | None:
    candidates = [p for p in runs.glob("live-*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class RunWatcher:
    """Turns new audit files into messages, and keeps the current payload."""

    def __init__(self, run_dir: Path | None, client_dir: Path, runs_root: Path,
                 *, follow_latest: bool):
        self.client_dir = client_dir
        self.runs_root = runs_root
        self.follow_latest = follow_latest
        self.lock = threading.Lock()
        self.listeners: list[queue.Queue] = []
        self.plans: dict[int, dict] = {}
        self.executions: dict[int, dict] = {}
        self.seen: dict[str, float] = {}
        self.run_dir = run_dir
        self.intrinsics = virtual_intrinsics(client_dir, {})[0]
        self.payload = self._empty(run_dir)
        self.state = "等待运行"
        if run_dir is not None:
            self._reset(run_dir)

    # ---- payload -----------------------------------------------------
    def _empty(self, run_dir: Path | None) -> dict:
        return {"run": run_dir.name if run_dir else "(等待 live_client 启动)",
                "cycles": [], "shaping": {"enabled": False}, "prefetch": 0}

    def _reset(self, run_dir: Path) -> None:
        """Load whatever the run already has; later polls only add what is new."""
        self.run_dir = run_dir
        self.plans.clear()
        self.executions.clear()
        self.seen.clear()
        result = run_dir / "result.json"
        data = json.loads(result.read_text()) if result.is_file() else {}
        self.intrinsics, label = virtual_intrinsics(self.client_dir, data)
        payload = self._empty(run_dir)
        payload["image_mode"] = label
        payload.update(self._meta(data))
        self._scan_dir(run_dir / "audit_plans", load_plan, self.plans)
        self._scan_dir(run_dir / "audit_executions", load_execution, self.executions)
        payload["cycles"] = [
            build_cycle(self.plans.get(key), self.executions.get(key), self.intrinsics)
            for key in sorted(set(self.plans) | set(self.executions))]
        with self.lock:
            self.payload = payload

    def _meta(self, result: dict) -> dict:
        planner = result.get("planner") or {}
        return {"model": planner.get("model"), "task": planner.get("task"),
                "contract": planner.get("contract"),
                "speed_ratio": planner.get("speed_ratio"),
                "deployment_id": planner.get("deployment_id"),
                "shaping": planner.get("shaping") or {"enabled": False},
                "anchor_mode": planner.get("anchor_mode"),
                "prefetch": (result.get("args") or {}).get("prefetch") or 0,
                "execute": result.get("execute"), "stop_reason": result.get("stop_reason"),
                "error": result.get("error")}

    # ---- fan-out -----------------------------------------------------
    def subscribe(self) -> queue.Queue:
        channel: queue.Queue = queue.Queue(maxsize=64)
        with self.lock:
            self.listeners.append(channel)
        return channel

    def unsubscribe(self, channel: queue.Queue) -> None:
        with self.lock:
            if channel in self.listeners:
                self.listeners.remove(channel)

    def broadcast(self, message: dict) -> None:
        with self.lock:
            listeners = list(self.listeners)
        for channel in listeners:
            try:
                channel.put_nowait(message)
            except queue.Full:                  # a stalled browser must not block us
                self.unsubscribe(channel)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.payload, ensure_ascii=False, allow_nan=False))

    # ---- polling -----------------------------------------------------
    def _changed(self, path: Path) -> bool:
        key = str(path)
        stamp = path.stat().st_mtime
        if self.seen.get(key) == stamp:
            return False
        self.seen[key] = stamp
        return True

    def _scan_dir(self, folder: Path, loader, store) -> set[int]:
        touched: set[int] = set()
        if not folder.is_dir():
            return touched
        for json_path in sorted(folder.glob("*.json")):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.is_file() or not self._changed(json_path):
                continue
            try:
                record = loader(npz_path)
            except (OSError, ValueError, KeyError):
                self.seen.pop(str(json_path), None)   # half-written; retry next poll
                continue
            store[int(record["meta"]["plan_id"])] = record
            touched.add(int(record["meta"]["plan_id"]))
        return touched

    def poll(self) -> None:
        if self.follow_latest:
            latest = newest_run(self.runs_root)
            if latest is not None and latest != self.run_dir:
                self._reset(latest)
                self.broadcast({"reload": True})
                return
        if self.run_dir is None or not self.run_dir.is_dir():
            return
        touched = self._scan_dir(self.run_dir / "audit_plans", load_plan, self.plans)
        touched |= self._scan_dir(self.run_dir / "audit_executions",
                                  load_execution, self.executions)
        for plan_id in sorted(touched):
            cycle = build_cycle(self.plans.get(plan_id), self.executions.get(plan_id),
                                self.intrinsics)
            with self.lock:
                cycles = self.payload.setdefault("cycles", [])
                for index, existing in enumerate(cycles):
                    if existing.get("plan_id") == plan_id:
                        cycles[index] = cycle
                        break
                else:
                    cycles.append(cycle)
            self.broadcast({"cycle": cycle})
        result = self.run_dir / "result.json"
        if result.is_file() and self._changed(result):
            try:
                meta = self._meta(json.loads(result.read_text()))
            except ValueError:
                self.seen.pop(str(result), None)
                return
            with self.lock:
                self.payload.update(meta)
            self.broadcast({"meta": meta, "state": "已结束"})
            self.state = "已结束"
        elif touched:
            self.state = "运行中"
            self.broadcast({"state": "运行中"})

    def run(self) -> None:
        while True:
            try:
                self.poll()
            except Exception as error:          # never let the watcher thread die
                print("monitor poll failed: %r" % error, file=sys.stderr, flush=True)
            time.sleep(POLL_S)


def make_handler(watcher: RunWatcher):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):          # keep the console for the robot
            pass

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                       # noqa: N802 (BaseHTTPRequestHandler API)
            if self.path in ("/", "/index.html"):
                html = page(watcher.snapshot(), LIVE_SCRIPT)
                self._send(html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/events"):
                self._events()
            elif self.path.startswith("/data"):
                body = json.dumps(watcher.snapshot(), ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            else:
                self.send_error(404)

        def _events(self):
            channel = watcher.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self._write({"state": watcher.state})
                while True:
                    try:
                        message = channel.get(timeout=KEEPALIVE_S)
                    except queue.Empty:
                        message = {"keepalive": True}
                    self._write(message)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                watcher.unsubscribe(channel)

        def _write(self, message: dict) -> None:
            body = json.dumps(message, ensure_ascii=False, allow_nan=False)
            self.wfile.write(("data: %s\n\n" % body).encode("utf-8"))
            self.wfile.flush()

    return Handler


def main() -> int:
    client = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="run directory; default: newest, and "
                                                 "switch to any newer one that appears")
    parser.add_argument("--runs-root", type=Path, default=client / "runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    run_dir = args.run
    follow_latest = run_dir is None
    if run_dir is None:
        run_dir = newest_run(args.runs_root)
    elif not run_dir.is_dir():
        print("no such run directory: %s" % run_dir, file=sys.stderr)
        return 1

    watcher = RunWatcher(run_dir, client, args.runs_root, follow_latest=follow_latest)
    threading.Thread(target=watcher.run, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(watcher))
    server.daemon_threads = True
    print("live monitor on http://%s:%d/  (run: %s)"
          % (args.host, args.port, run_dir.name if run_dir else "waiting"), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

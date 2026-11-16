import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
import urllib.request

from http.server import ThreadingHTTPServer

from live_monitor import LIVE_SCRIPT, RunWatcher, make_handler, newest_run
from test_visualize_run import write_run

HERE = Path(__file__).resolve().parent


def staged(root: Path):
    """A finished run plus a pristine copy of its audit files to replay from."""
    full = write_run(root / "source")
    live = root / "runs" / "live-exec-20261117-000000-000000"
    (live / "audit_plans").mkdir(parents=True)
    (live / "audit_executions").mkdir(parents=True)
    return full, live


def copy_cycle(full: Path, live: Path, *, plan=True, execution=True):
    if plan:
        for name in ("plan0000.npz", "plan0000.json"):
            shutil.copy2(full / "audit_plans" / name, live / "audit_plans" / name)
    if execution:
        for name in ("exec0000.npz", "exec0000.json"):
            shutil.copy2(full / "audit_executions" / name, live / "audit_executions" / name)


class WatcherTests(unittest.TestCase):
    def test_empty_run_serves_a_page_and_fills_in_later(self):
        with tempfile.TemporaryDirectory() as folder:
            full, live = staged(Path(folder))
            watcher = RunWatcher(live, HERE, live.parent, follow_latest=False)
            self.assertEqual(watcher.snapshot()["cycles"], [])

            channel = watcher.subscribe()
            copy_cycle(full, live, execution=False)
            watcher.poll()
            message = channel.get_nowait()
            self.assertEqual(message["cycle"]["plan_id"], 0)
            self.assertNotIn("feedback", message["cycle"])
            self.assertEqual(len(watcher.snapshot()["cycles"]), 1)

            # The execution record for the same plan replaces the cycle in place.
            copy_cycle(full, live, plan=False)
            watcher.poll()
            while True:
                message = channel.get_nowait()
                if "cycle" in message:
                    break
            self.assertEqual(len(watcher.snapshot()["cycles"]), 1)
            cycle = watcher.snapshot()["cycles"][0]
            self.assertIn("feedback", cycle)
            self.assertIsNotNone(cycle["image"])        # plan evidence is kept
            self.assertTrue(cycle["executed"])

    def test_poll_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            full, live = staged(Path(folder))
            copy_cycle(full, live)
            watcher = RunWatcher(live, HERE, live.parent, follow_latest=False)
            self.assertEqual(len(watcher.snapshot()["cycles"]), 1)
            channel = watcher.subscribe()
            watcher.poll()
            watcher.poll()
            self.assertEqual(len(watcher.snapshot()["cycles"]), 1)
            remaining = [channel.get_nowait() for _ in range(channel.qsize())]
            self.assertFalse([m for m in remaining if "cycle" in m])

    def test_result_json_updates_the_header(self):
        with tempfile.TemporaryDirectory() as folder:
            full, live = staged(Path(folder))
            watcher = RunWatcher(live, HERE, live.parent, follow_latest=False)
            channel = watcher.subscribe()
            shutil.copy2(full / "result.json", live / "result.json")
            watcher.poll()
            messages = [channel.get_nowait() for _ in range(channel.qsize())]
            meta = [m for m in messages if "meta" in m][-1]
            self.assertEqual(meta["meta"]["task"], "tidying")
            self.assertEqual(meta["state"], "已结束")
            self.assertEqual(watcher.snapshot()["prefetch"], 0.5)

    def test_follow_latest_switches_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            full, live = staged(Path(folder))
            watcher = RunWatcher(live, HERE, live.parent, follow_latest=True)
            channel = watcher.subscribe()
            newer = live.parent / "live-exec-20261117-111111-111111"
            (newer / "audit_plans").mkdir(parents=True)
            watcher.poll()
            self.assertEqual(watcher.run_dir, newer)
            self.assertTrue(any(m.get("reload") for m in
                                [channel.get_nowait() for _ in range(channel.qsize())]))
            self.assertEqual(newest_run(live.parent), newer)

    def test_half_written_record_is_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            full, live = staged(Path(folder))
            watcher = RunWatcher(live, HERE, live.parent, follow_latest=False)
            (live / "audit_plans" / "plan0000.json").write_text('{"plan_id": 0')  # truncated
            shutil.copy2(full / "audit_plans/plan0000.npz", live / "audit_plans/plan0000.npz")
            watcher.poll()
            self.assertEqual(watcher.snapshot()["cycles"], [])
            shutil.copy2(full / "audit_plans/plan0000.json",
                         live / "audit_plans/plan0000.json")
            watcher.poll()
            self.assertEqual(len(watcher.snapshot()["cycles"]), 1)


class ServerTests(unittest.TestCase):
    def test_http_serves_page_data_and_events(self):
        with tempfile.TemporaryDirectory() as folder:
            full, live = staged(Path(folder))
            copy_cycle(full, live)
            watcher = RunWatcher(live, HERE, live.parent, follow_latest=False)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(watcher))
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = "http://127.0.0.1:%d" % server.server_address[1]
            try:
                html = urllib.request.urlopen(base + "/").read().decode()
                self.assertIn("const DATA = {", html)
                self.assertIn("new EventSource('events')", html)
                self.assertNotIn("__LIVE__", html)
                self.assertNotIn("NaN", html)

                payload = json.loads(urllib.request.urlopen(base + "/data").read())
                self.assertEqual(len(payload["cycles"]), 1)

                stream = urllib.request.urlopen(base + "/events")
                self.assertTrue(stream.headers["Content-Type"].startswith("text/event-stream"))
                first = json.loads(stream.readline().decode().split("data: ", 1)[1])
                self.assertIn("state", first)
                stream.close()
            finally:
                server.shutdown()
                server.server_close()

    def test_live_script_only_touches_existing_hooks(self):
        for name in ("renderMeta", "buildTable", "select", "DATA.cycles"):
            self.assertIn(name, LIVE_SCRIPT)


if __name__ == "__main__":
    unittest.main()

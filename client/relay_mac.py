#!/usr/bin/env python3
"""Run the two loopback-only SSH forwards required between b300, Mac and zhuji."""
from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import time


def relay_commands(server_host, server_port=18765, relay_port=28765,
                   b300_alias="b300", zhuji_alias="zhuji"):
    if not server_host or not 1 <= int(server_port) <= 65535 or not 1 <= int(relay_port) <= 65535:
        raise ValueError("invalid relay endpoint")
    common = ["ssh", "-N", "-T", "-o", "ExitOnForwardFailure=yes",
              "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
    local = common + ["-L", "127.0.0.1:%d:%s:%d" %
                      (int(relay_port), server_host, int(server_port)), b300_alias]
    reverse = common + ["-R", "127.0.0.1:%d:127.0.0.1:%d" %
                        (int(relay_port), int(relay_port)), zhuji_alias]
    return local, reverse


def wait_local(port, processes, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("SSH relay exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("local XFlow relay port did not open")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", required=True,
                        help="current pod host from the current endpoint.json")
    parser.add_argument("--server-port", type=int, default=18765)
    parser.add_argument("--relay-port", type=int, default=28765)
    parser.add_argument("--b300-alias", default="b300")
    parser.add_argument("--zhuji-alias", default="zhuji")
    args = parser.parse_args()
    commands = relay_commands(args.server_host, args.server_port, args.relay_port,
                              args.b300_alias, args.zhuji_alias)
    # SIGTERM/SIGHUP must run the cleanup below, otherwise the ssh children are
    # orphaned and keep holding port 28765 on the Mac and on zhuji.
    def terminate(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    processes = [subprocess.Popen(command) for command in commands]
    try:
        wait_local(args.relay_port, processes)
        print("relay ready on zhuji 127.0.0.1:%d; Ctrl-C to close" % args.relay_port,
              flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("one SSH relay exited")
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run the two loopback-only SSH forwards required between b300, Mac and zhuji."""
from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import time


def relay_commands(routes, b300_alias="b300", zhuji_alias="zhuji"):
    """One ssh to b300 and one to zhuji, each carrying every route.

    ``routes`` is a list of ``(server_host, server_port, relay_port)``; a pod
    that serves several checkpoints gets one local port per checkpoint, so the
    robot can switch between them with ``--relay-port``.
    """
    routes = [(str(host), int(server), int(local)) for host, server, local in routes]
    if not routes:
        raise ValueError("at least one route is required")
    if len({local for _, _, local in routes}) != len(routes):
        raise ValueError("relay ports must be unique")
    for host, server, local in routes:
        if not host or not 1 <= server <= 65535 or not 1 <= local <= 65535:
            raise ValueError("invalid relay endpoint")
    common = ["ssh", "-N", "-T", "-o", "ExitOnForwardFailure=yes",
              "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
    local_cmd = list(common)
    reverse_cmd = list(common)
    for host, server, port in routes:
        local_cmd += ["-L", "127.0.0.1:%d:%s:%d" % (port, host, server)]
        reverse_cmd += ["-R", "127.0.0.1:%d:127.0.0.1:%d" % (port, port)]
    return local_cmd + [b300_alias], reverse_cmd + [zhuji_alias]


def wait_local(ports, processes, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    pending = list(ports)
    while pending and time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("SSH relay exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", pending[0]), timeout=0.25):
                pending.pop(0)
                continue
        except OSError:
            time.sleep(0.1)
    if pending:
        raise TimeoutError("local XFlow relay port %d did not open" % pending[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", help="pod host from the current endpoint.json")
    parser.add_argument("--server-port", type=int, default=18765)
    parser.add_argument("--relay-port", type=int, default=28765)
    parser.add_argument("--route", action="append", default=[], metavar="HOST:SRV:LOCAL",
                        help="repeatable; forwards zhuji 127.0.0.1:LOCAL to HOST:SRV")
    parser.add_argument("--b300-alias", default="b300")
    parser.add_argument("--zhuji-alias", default="zhuji")
    args = parser.parse_args()
    if args.route:
        routes = []
        for item in args.route:
            host, server, local = item.rsplit(":", 2)
            routes.append((host, int(server), int(local)))
    elif args.server_host:
        routes = [(args.server_host, args.server_port, args.relay_port)]
    else:
        parser.error("give --server-host or at least one --route")
    commands = relay_commands(routes, args.b300_alias, args.zhuji_alias)
    # SIGTERM/SIGHUP must run the cleanup below, otherwise the ssh children are
    # orphaned and keep holding port 28765 on the Mac and on zhuji.
    def terminate(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    processes = [subprocess.Popen(command) for command in commands]
    try:
        wait_local([local for _, _, local in routes], processes)
        for host, server, local in routes:
            print("  zhuji 127.0.0.1:%d  ->  %s:%d" % (local, host, server), flush=True)
        print("relay ready (%d route%s); Ctrl-C to close"
              % (len(routes), "" if len(routes) == 1 else "s"), flush=True)
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


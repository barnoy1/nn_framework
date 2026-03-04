from __future__ import annotations

import os
import signal
import socket
import subprocess
import time


def is_local_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def wait_for_service(host: str, port: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                if sock.connect_ex((host, int(port))) == 0:
                    return True
            except OSError:
                pass
        time.sleep(0.1)
    return False


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            return sock.connect_ex((host, int(port))) == 0
        except OSError:
            return False


def find_available_port(host: str, start_port: int, attempts: int = 30) -> int:
    base = int(start_port)
    for offset in range(max(1, int(attempts))):
        candidate = base + offset
        if not is_port_in_use(host, candidate):
            return candidate
    return base


def pids_on_port(port: int) -> list[int]:
    discovered: list[int] = []

    lsof = subprocess.run(
        ["lsof", "-ti", f"tcp:{int(port)}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if lsof.returncode == 0 and lsof.stdout:
        for line in lsof.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                discovered.append(int(line))

    if discovered:
        return sorted(set(discovered))

    fuser = subprocess.run(
        ["fuser", "-n", "tcp", str(int(port))],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (fuser.stdout or "") + " " + (fuser.stderr or "")
    for token in output.split():
        token = token.strip()
        if token.isdigit():
            discovered.append(int(token))

    return sorted(set(discovered))


def free_port_for_reuse(host: str, port: int, logger_port) -> bool:
    if not is_local_host(host):
        logger_port.warning("Port cleanup skipped for non-local host {}:{}", host, int(port))
        return False

    pids = [pid for pid in pids_on_port(int(port)) if pid != os.getpid()]
    if not pids:
        return not is_port_in_use(host, int(port))

    logger_port.warning("Port {} is occupied by pid(s) {}; terminating to reuse configured port.", int(port), pids)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger_port.warning("No permission to terminate pid {} on port {}", pid, int(port))

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not is_port_in_use(host, int(port)):
            return True
        time.sleep(0.1)

    remaining = [pid for pid in pids_on_port(int(port)) if pid != os.getpid()]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger_port.warning("No permission to force-kill pid {} on port {}", pid, int(port))

    return not is_port_in_use(host, int(port))

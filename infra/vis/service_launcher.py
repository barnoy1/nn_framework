from __future__ import annotations

import os
from pathlib import Path
import signal
from typing import Dict
import socket
import subprocess
import sys
import time


_SERVICE_PROCESSES: Dict[str, subprocess.Popen] = {}


def _is_local_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def _wait_for_service(host: str, port: int, timeout_seconds: float = 5.0) -> bool:
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


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            return sock.connect_ex((host, int(port))) == 0
        except OSError:
            return False


def _find_available_port(host: str, start_port: int, attempts: int = 30) -> int:
    base = int(start_port)
    for offset in range(max(1, int(attempts))):
        candidate = base + offset
        if not _is_port_in_use(host, candidate):
            return candidate
    return base


def _pids_on_port(port: int) -> list[int]:
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


def _free_port_for_reuse(host: str, port: int, logger_port) -> bool:
    if not _is_local_host(host):
        logger_port.warning("Port cleanup skipped for non-local host {}:{}", host, int(port))
        return False

    pids = [pid for pid in _pids_on_port(int(port)) if pid != os.getpid()]
    if not pids:
        return not _is_port_in_use(host, int(port))

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
        if not _is_port_in_use(host, int(port)):
            return True
        time.sleep(0.1)

    remaining = [pid for pid in _pids_on_port(int(port)) if pid != os.getpid()]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger_port.warning("No permission to force-kill pid {} on port {}", pid, int(port))

    return not _is_port_in_use(host, int(port))


def _read_log_tail(path: Path, lines: int = 8) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if not content:
        return ""
    return " | ".join(content[-lines:])


def _is_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def start_mlflow_ui_service(
    *,
    tracking_dir: Path,
    host: str,
    port: int,
    logger_port,
    tracking_backend: str = "sqlite",
    sqlite_db_name: str = "mlflow.db",
) -> str:
    resolved_tracking_dir = tracking_dir.resolve()
    backend = str(tracking_backend).strip().lower()
    requested_port = int(port)
    selected_port = requested_port
    if _is_port_in_use(host, requested_port):
        if not _free_port_for_reuse(host, requested_port, logger_port):
            selected_port = _find_available_port(host, requested_port)
            if selected_port != requested_port:
                logger_port.warning(
                    "MLflow UI port {} still in use; using {} instead.",
                    requested_port,
                    selected_port,
                )

    url = f"http://{host}:{selected_port}"
    service_key = f"mlflow:{host}:{selected_port}:{resolved_tracking_dir}:{backend}:{sqlite_db_name}"
    existing = _SERVICE_PROCESSES.get(service_key)
    if _is_running(existing):
        logger_port.info("MLflow UI is up: {}", url)
        return url

    if backend == "sqlite":
        sqlite_path = (resolved_tracking_dir / str(sqlite_db_name)).resolve()
        backend_store_uri = f"sqlite:///{sqlite_path}"
    else:
        backend_store_uri = resolved_tracking_dir.as_uri()

    service_log = resolved_tracking_dir / "mlflow_ui.log"
    log_handle = service_log.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        resolved_tracking_dir.as_uri(),
        "--host",
        str(host),
        "--port",
        str(selected_port),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        cwd=str(resolved_tracking_dir),
    )
    _SERVICE_PROCESSES[service_key] = process

    if _wait_for_service(host=host, port=selected_port, timeout_seconds=15.0):
        logger_port.info("MLflow UI is up: {}", url)
        logger_port.info("MLflow tracking store: {} ({})", resolved_tracking_dir, backend)
    elif process.poll() is not None:
        tail = _read_log_tail(service_log)
        if tail:
            logger_port.warning("MLflow UI failed to start: {}", tail)
        logger_port.warning("MLflow UI failed to start. Check {}", service_log)
    else:
        logger_port.info("MLflow UI starting: {}", url)
        logger_port.info("MLflow UI logs: {}", service_log)
    return url


def start_tensorboard_service(*, log_dir: Path, host: str, port: int, logger_port) -> str:
    resolved_log_dir = log_dir.resolve()
    requested_port = int(port)
    selected_port = requested_port
    if _is_port_in_use(host, requested_port):
        if not _free_port_for_reuse(host, requested_port, logger_port):
            selected_port = _find_available_port(host, requested_port)
            if selected_port != requested_port:
                logger_port.warning(
                    "TensorBoard port {} still in use; using {} instead.",
                    requested_port,
                    selected_port,
                )

    url = f"http://{host}:{selected_port}"
    service_key = f"tensorboard:{host}:{selected_port}:{resolved_log_dir}"
    existing = _SERVICE_PROCESSES.get(service_key)
    if _is_running(existing):
        logger_port.info("TensorBoard is up: {}", url)
        return url

    service_log = resolved_log_dir / "tensorboard.log"
    log_handle = service_log.open("a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(resolved_log_dir),
        "--host",
        str(host),
        "--port",
        str(selected_port),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        cwd=str(resolved_log_dir),
    )
    _SERVICE_PROCESSES[service_key] = process

    if _wait_for_service(host=host, port=selected_port, timeout_seconds=15.0):
        logger_port.info("TensorBoard is up: {}", url)
        logger_port.info("TensorBoard logdir: {}", resolved_log_dir)
    elif process.poll() is not None:
        tail = _read_log_tail(service_log)
        if tail:
            logger_port.warning("TensorBoard failed to start: {}", tail)
            if "pkg_resources" in tail:
                logger_port.warning("TensorBoard requires setuptools (pkg_resources). Install with: pip install setuptools")
        logger_port.warning("TensorBoard failed to start. Check {}", service_log)
    else:
        logger_port.info("TensorBoard starting: {}", url)
        logger_port.info("TensorBoard service logs: {}", service_log)
    return url

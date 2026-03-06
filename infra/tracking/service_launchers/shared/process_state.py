from __future__ import annotations

import subprocess
from typing import Dict


SERVICE_PROCESSES: Dict[str, subprocess.Popen] = {}


def is_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def get_running_process(service_key: str) -> subprocess.Popen | None:
    process = SERVICE_PROCESSES.get(service_key)
    return process if is_running(process) else None


def register_process(service_key: str, process: subprocess.Popen) -> None:
    SERVICE_PROCESSES[service_key] = process

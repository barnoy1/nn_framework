#!/usr/bin/env sh
set -eu

if [ "${ENABLE_DEBUGPY:-0}" = "1" ]; then
  debugpy_host="${DEBUGPY_HOST:-0.0.0.0}"
  debugpy_port="${DEBUGPY_PORT:-5678}"
  if [ "${DEBUGPY_WAIT_FOR_CLIENT:-1}" = "1" ]; then
    exec python -m debugpy --listen "${debugpy_host}:${debugpy_port}" --wait-for-client cli.py "$@"
  fi
  exec python -m debugpy --listen "${debugpy_host}:${debugpy_port}" cli.py "$@"
fi

exec python cli.py "$@"
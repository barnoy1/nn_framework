#!/usr/bin/env bash
set -euo pipefail

TARGET_PATH="$PWD"
USER_DATA_DIR="${VSCODE_USER_DATA_DIR:-/tmp/vscode-root}"

echo "[code-root] user=$(whoami)"
echo "[code-root] pwd=$(pwd)"
echo "[code-root] DISPLAY=${DISPLAY:-<unset>}"
echo "[code-root] XAUTHORITY=${XAUTHORITY:-<unset>}"
echo "[code-root] DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-<unset>}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[code-root] error: DISPLAY is not set; GUI forwarding is unavailable." >&2
  exit 1
fi

if [[ -n "${XAUTHORITY:-}" && ! -e "${XAUTHORITY}" ]]; then
  echo "[code-root] warning: XAUTHORITY points to missing file: ${XAUTHORITY}" >&2
fi

echo "[code-root] launching: code --no-sandbox --user-data-dir ${USER_DATA_DIR} ${TARGET_PATH}"
exec code --no-sandbox --user-data-dir "${USER_DATA_DIR}" "${TARGET_PATH}"

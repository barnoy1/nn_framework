#!/usr/bin/env sh
set -eu

if [ -d "/workspace/scripts" ]; then
	chmod +x /workspace/scripts/*.sh 2>/dev/null || true
	chmod +x /workspace/scripts/gui/*.sh 2>/dev/null || true
	chmod +x /workspace/scripts/utilities/*.sh 2>/dev/null || true
fi

echo "[runner_entrypoint] /dev/shm usage"
df -h /dev/shm || true

echo "[runner_entrypoint] shm mounts"
mount | grep shm || echo "[runner_entrypoint] no shm mounts reported"

exec /bin/bash

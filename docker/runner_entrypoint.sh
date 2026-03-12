#!/usr/bin/env sh
set -eu

if [ -d "/workspace/scripts" ]; then
	chmod +x /workspace/scripts/*.sh 2>/dev/null || true
	chmod +x /workspace/scripts/gui/*.sh 2>/dev/null || true
fi

exec /bin/bash

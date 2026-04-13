#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not available." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Error: docker compose plugin is not available." >&2
  exit 1
fi

print_usage() {
  cat <<'EOF'
Usage: docker/profiles/u22-cu128/build.sh [options]

Build the Ubuntu 22.04 + CUDA 12.8 runner profile:
  docker compose -f docker/profiles/u22-cu128/docker-compose.yml build runner

Options:
  -h, --help       Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

docker compose \
  -f docker/profiles/u22-cu128/docker-compose.yml \
  build runner

echo "Built runner profile image: nn-framework-runner:local"

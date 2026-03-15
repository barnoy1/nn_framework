#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace"
PYTHON_BIN="${NN_PYTHON_BIN:-python}"

CONFIG_PATH="${NN_EXPERIMENT_CONFIG:-}"
OUTPUT_DIR="${NN_OUTPUT_DIR:-}"
MODEL_SOURCE_ROOT="${NN_MODEL_SOURCE_ROOT:-}"

has_arg() {
  local flag="$1"
  shift
  for token in "$@"; do
    if [[ "$token" == "$flag" || "$token" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

has_source_root_override() {
  for token in "$@"; do
    if [[ "$token" == *"model.source_root="* ]]; then
      return 0
    fi
  done
  return 1
}

print_usage() {
  cat <<'EOF'
  Usage: /workspace/scripts/train.sh [CLI train args]

Runs inside the container and forwards args to:
  python cli.py train

  Working directory is set to /workspace before execution.

  Default settings:
    /workspace/scripts/train.sh

Example (inside docker):
    /workspace/scripts/train.sh \
      --config experiments/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yaml \
    --output-dir /workspace/out

Defaults come from NN_* env vars exposed inside the container.
EOF
}

if has_arg "-h" "$@" || has_arg "--help" "$@"; then
  print_usage
  exit 0
fi

cd "$REPO_ROOT"

CMD=("$PYTHON_BIN" cli.py train)

if ! has_arg "--config" "$@"; then
  if [[ -n "$CONFIG_PATH" ]]; then
    CMD+=(--config "$CONFIG_PATH")
  fi
fi
if ! has_arg "--output-dir" "$@"; then
  if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
  fi
fi
if [[ -n "$MODEL_SOURCE_ROOT" ]] && ! has_source_root_override "$@"; then
  CMD+=(--overrides "model.source_root=$MODEL_SOURCE_ROOT")
fi
CMD+=("$@")

exec "${CMD[@]}"

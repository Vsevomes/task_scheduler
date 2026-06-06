#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${STARPU_DOCKER_IMAGE:-starpu-cuda-rtx4060}"
STARPU_HOME="${STARPU_HOME:-/workspace/.starpu-home}"

GPU_ARGS=()
if [[ "${USE_GPU:-0}" == "1" ]]; then
  GPU_ARGS=(--gpus all)
fi

mkdir -p "$ROOT/.starpu-home"

docker run --rm "${GPU_ARGS[@]}" \
  -v "$ROOT:/workspace" \
  -w /workspace/build \
  -u "$(id -u):$(id -g)" \
  -e "STARPU_HOME=$STARPU_HOME" \
  "$IMAGE" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${STARPU_DOCKER_IMAGE:-starpu-cuda-rtx4060}"

docker build -t "$IMAGE" -f "$ROOT/docker_rtx4060/Dockerfile" "$ROOT/docker_rtx4060"
echo "Built image: $IMAGE"

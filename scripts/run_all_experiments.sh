#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run_experiment.sh"

MODES=(native_cpu native_gpu starpu_hybrid)
STARPU_MODES=(starpu_hybrid)
MATMUL_SIZES=(512 1024 2048)
IMAGE_SIZES=("640 480" "1280 720")
TASK_COUNTS=(100 1000)

echo "=== hello ==="
for mode in "${STARPU_MODES[@]}"; do
  "$RUN" hello --mode "$mode" --size 1048576
done

echo "=== matmul ==="
for mode in "${MODES[@]}"; do
  for size in "${MATMUL_SIZES[@]}"; do
    "$RUN" matmul --mode "$mode" --size "$size"
  done
done

echo "=== independent ==="
for mode in "${STARPU_MODES[@]}"; do
  for tasks in "${TASK_COUNTS[@]}"; do
    "$RUN" independent --mode "$mode" --tasks "$tasks" --size 256
  done
done

echo "=== heterogeneous ==="
for mode in "${STARPU_MODES[@]}"; do
  "$RUN" heterogeneous --mode "$mode" --tasks 100 --light-ratio 0.7 --light-size 256 --heavy-size 384
done

echo "=== image ==="
for mode in "${MODES[@]}"; do
  for dims in "${IMAGE_SIZES[@]}"; do
    read -r w h <<<"$dims"
    for op in grayscale blur threshold; do
      "$RUN" image --mode "$mode" --width "$w" --height "$h" --op "$op"
    done
  done
done

echo "=== overhead ==="
for mode in "${STARPU_MODES[@]}"; do
  "$RUN" overhead --mode "$mode" --kind noop --tasks 5000
  "$RUN" overhead --mode "$mode" --kind memcpy --tasks 500 --bytes 1048576
done

echo "All experiments finished. Results in results/"

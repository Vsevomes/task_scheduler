#!/usr/bin/env bash
# Report experiment matrix (extended parameters for thesis benchmarks).
#
# Usage:
#   ./scripts/run_report_experiments.sh              # full matrix
#   ./scripts/run_report_experiments.sh matmul       # one scenario
#   STARPU_PROF=0 ./scripts/run_report_experiments.sh
#
# StarPU hybrid runs use STARPU_SCHED=dmda (override with STARPU_SCHED=...).
# First dmda run per codelet may calibrate perfmodels (.starpu-home/).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run_experiment.sh"
LOG="${REPORT_LOG:-$ROOT/results/report_run.log}"

MODES=(native_cpu native_gpu starpu_hybrid)

# --- matmul: matrix sizes N×N, fixed tile ---
MATMUL_SIZES=(512 2048 8192)
MATMUL_TILE=256

# --- independent: number of blocks × block size ---
INDEPENDENT_TASKS=(1500 9000 15000)
INDEPENDENT_BLOCK=4096

# --- heterogeneous: task count + light / medium / heavy mix ---
HETERO_TASKS=(3000 9000 15000)
HETERO_LIGHT_RATIO=0.5   # 50% light tasks
HETERO_MEDIUM_RATIO=0.3  # 30% medium tasks
# remaining 20% heavy
HETERO_LIGHT_SIZE=2048    # small arrays, low arithmetic intensity
HETERO_MEDIUM_SIZE=16384  # medium arrays
HETERO_HEAVY_SIZE=65536   # large arrays, GPU-friendly

# --- image: UHD resolutions, blur + mixed tile ops (StarPU report case) ---
IMAGE_SIZES=(
  "7680 4320"    # 8K UHD  (~33 MP)
  "15360 8640"   # 16K     (~133 MP)
  "30720 17280"  # 32K     (~531 MP)
)
IMAGE_OP=blur
IMAGE_TILE=64

mkdir -p "$ROOT/results/report"
exec > >(tee -a "$LOG") 2>&1

starpu_env() {
  if [[ "${1:-}" == starpu_hybrid ]]; then
    export STARPU_SCHED="${STARPU_SCHED:-dmda}"
  else
    unset STARPU_SCHED 2>/dev/null || true
  fi
  export STARPU_PROF="${STARPU_PROF:-0}"
}

run_matmul() {
  echo "=== matmul (sizes: ${MATMUL_SIZES[*]}, tile=${MATMUL_TILE}) ==="
  for mode in "${MODES[@]}"; do
    starpu_env "$mode"
    for size in "${MATMUL_SIZES[@]}"; do
      echo "--- matmul mode=$mode size=$size ---"
      "$RUN" matmul --mode "$mode" --size "$size" --tile-size "$MATMUL_TILE" \
        --output "$ROOT/results/report/matmul_${mode}_${size}.json"
    done
  done
}

run_independent() {
  echo "=== independent (tasks: ${INDEPENDENT_TASKS[*]}, block=${INDEPENDENT_BLOCK}) ==="
  for mode in "${MODES[@]}"; do
    starpu_env "$mode"
    for tasks in "${INDEPENDENT_TASKS[@]}"; do
      echo "--- independent mode=$mode tasks=$tasks ---"
      "$RUN" independent --mode "$mode" --tasks "$tasks" --size "$INDEPENDENT_BLOCK" \
        --output "$ROOT/results/report/independent_${mode}_${tasks}.json"
    done
  done
}

run_heterogeneous() {
  echo "=== heterogeneous (tasks: ${HETERO_TASKS[*]}) ==="
  echo "    mix: light=${HETERO_LIGHT_RATIO} medium=${HETERO_MEDIUM_RATIO} heavy=0.2"
  echo "    sizes: light=${HETERO_LIGHT_SIZE} medium=${HETERO_MEDIUM_SIZE} heavy=${HETERO_HEAVY_SIZE}"
  for mode in "${MODES[@]}"; do
    starpu_env "$mode"
    for tasks in "${HETERO_TASKS[@]}"; do
      echo "--- heterogeneous mode=$mode tasks=$tasks ---"
      "$RUN" heterogeneous --mode "$mode" --tasks "$tasks" \
        --light-ratio "$HETERO_LIGHT_RATIO" --medium-ratio "$HETERO_MEDIUM_RATIO" \
        --light-size "$HETERO_LIGHT_SIZE" --medium-size "$HETERO_MEDIUM_SIZE" \
        --heavy-size "$HETERO_HEAVY_SIZE" \
        --output "$ROOT/results/report/heterogeneous_${mode}_${tasks}.json"
    done
  done
}

run_image() {
  echo "=== image (op=${IMAGE_OP}, tile=${IMAGE_TILE}, mixed-ops) ==="
  for mode in "${MODES[@]}"; do
    starpu_env "$mode"
    for dims in "${IMAGE_SIZES[@]}"; do
      read -r w h <<<"$dims"
      label="${w}x${h}"
      echo "--- image mode=$mode ${label} ---"
      "$RUN" image --mode "$mode" --width "$w" --height "$h" \
        --op "$IMAGE_OP" --tile-size "$IMAGE_TILE" --mixed-ops \
        --output "$ROOT/results/report/image_${mode}_${label}.json"
    done
  done
}

print_matrix() {
  cat <<EOF

Report matrix summary
=====================
Modes (each scenario): ${MODES[*]}
StarPU scheduler: \${STARPU_SCHED:-dmda}

matmul       N=${MATMUL_SIZES[*]}  tile=${MATMUL_TILE}  -> 9 runs
independent  tasks=${INDEPENDENT_TASKS[*]}  block=${INDEPENDENT_BLOCK}  -> 9 runs
heterogeneous tasks=${HETERO_TASKS[*]}
              light ${HETERO_LIGHT_RATIO} / medium ${HETERO_MEDIUM_RATIO} / heavy 0.2
              sizes ${HETERO_LIGHT_SIZE} / ${HETERO_MEDIUM_SIZE} / ${HETERO_HEAVY_SIZE}  -> 9 runs
image        ${IMAGE_SIZES[*]}  op=${IMAGE_OP} mixed-ops  -> 9 runs

Total: 36 benchmark runs
Results: results/report/*.json
Log: $LOG

Note: matmul 8192 native_cpu and image 32K may take a long time.

EOF
}

ONLY="${1:-all}"
print_matrix
echo "Started $(date)"

case "$ONLY" in
  all)
    run_matmul
    run_independent
    run_heterogeneous
    run_image
    ;;
  matmul) run_matmul ;;
  independent) run_independent ;;
  heterogeneous) run_heterogeneous ;;
  image) run_image ;;
  *)
    echo "Unknown scenario: $ONLY (use all|matmul|independent|heterogeneous|image)"
    exit 1
    ;;
esac

echo "Finished $(date)"
echo "JSON count: $(find "$ROOT/results/report" -name '*.json' 2>/dev/null | wc -l)"

#!/usr/bin/env bash
# Compare StarPU schedulers on one starpu_hybrid run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/scripts/run_experiment.sh"

SCENARIO="${1:-heterogeneous}"
MODE="${MODE:-starpu_hybrid}"
SCHEDULERS=(lws dmda dmdas)

shift || true
BENCH_ARGS=("$@")
if [[ ${#BENCH_ARGS[@]} -eq 0 ]]; then
  case "$SCENARIO" in
    matmul)
      BENCH_ARGS=(--size 2048 --tile-size 256)
      ;;
    image)
      BENCH_ARGS=(--width 1280 --height 720 --tile-size 64 --op blur --mixed-ops)
      ;;
    heterogeneous)
      BENCH_ARGS=(--tasks 900 --light-ratio 0.5 --medium-ratio 0.3 --light-size 1024 --medium-size 4096 --heavy-size 16384)
      ;;
    *)
      BENCH_ARGS=()
      ;;
  esac
fi

OUT_DIR="$ROOT/results/scheduler_compare/${SCENARIO}"
mkdir -p "$OUT_DIR"

echo "Scenario: $SCENARIO  mode: $MODE  args: ${BENCH_ARGS[*]}"
echo "Output dir: $OUT_DIR"
echo ""

for sched in "${SCHEDULERS[@]}"; do
  export STARPU_SCHED="$sched"
  export STARPU_PROF=0
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="$OUT_DIR/${sched}_${stamp}.json"
  echo "=== STARPU_SCHED=$sched ==="
  "$RUN" "$SCENARIO" --mode "$MODE" "${BENCH_ARGS[@]}" --output "$out"
  echo ""
done

echo "Done. Results in $OUT_DIR"

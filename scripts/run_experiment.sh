#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

SCENARIO="${1:-}"
shift || true

if [[ -z "$SCENARIO" ]]; then
  echo "Usage: $0 SCENARIO [--mode MODE] [benchmark args...]"
  echo "Scenarios: hello matmul independent heterogeneous image overhead"
  exit 1
}

case "$SCENARIO" in
  hello) BIN=hello_starpu ;;
  matmul) BIN=bench_matmul ;;
  independent) BIN=bench_independent ;;
  heterogeneous) BIN=bench_heterogeneous ;;
  image) BIN=bench_image ;;
  overhead) BIN=bench_overhead ;;
  *)
    echo "Unknown scenario: $SCENARIO"
    exit 1
    ;;
esac

EXEC="$BUILD_DIR/$BIN"
if [[ ! -x "$EXEC" ]]; then
  echo "Binary not found: $EXEC (run ./starpu-rtx4060-build first)"
  exit 1
fi

METRICS_FILE="$ROOT/results/system_${SCENARIO}_$(date +%Y%m%d_%H%M%S).csv"
"$ROOT/scripts/collect_metrics.sh" 1 "$METRICS_FILE" &
SAMPLER_PID=$!

cleanup() {
  kill "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$ROOT"
export STARPU_PROF="${STARPU_PROF:-1}"
export STARPU_FXT_PREFIX="${STARPU_FXT_PREFIX:-results/trace_${SCENARIO}_}"

echo "Running $BIN with args: $*"
echo "System metrics -> $METRICS_FILE"
"$EXEC" "$@"

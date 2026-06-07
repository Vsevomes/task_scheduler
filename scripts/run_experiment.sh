#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

SCENARIO="${1:-}"
shift || true

if [[ -z "$SCENARIO" ]]; then
  echo "Usage: $0 SCENARIO [--mode MODE] [benchmark args...]"
  echo "Scenarios: matmul independent heterogeneous image overhead"
  exit 1
fi

case "$SCENARIO" in
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
METRICS_SAMPLE_SEC=0.25 "$ROOT/scripts/collect_metrics.sh" 0.5 "$METRICS_FILE" &
SAMPLER_PID=$!

cleanup() {
  kill "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$ROOT"
export STARPU_PROF="${STARPU_PROF:-1}"
export STARPU_FXT_PREFIX="${STARPU_FXT_PREFIX:-results/trace_${SCENARIO}_}"
export SYSTEM_METRICS_FILE="$METRICS_FILE"

echo "Running $BIN with args: $*"
echo "System metrics -> $METRICS_FILE"
BENCH_LOG="$(mktemp)"
trap 'kill "$SAMPLER_PID" 2>/dev/null || true; rm -f "$BENCH_LOG"' EXIT
"$EXEC" "$@" 2>&1 | tee "$BENCH_LOG"
JSON_PATH="$(grep -oE 'results/[^ ]+\.json' "$BENCH_LOG" | tail -1 || true)"
if [[ -n "$JSON_PATH" && ! "$JSON_PATH" = /* ]]; then
  JSON_PATH="$ROOT/$JSON_PATH"
fi
"$ROOT/scripts/summarize_metrics.sh" "$METRICS_FILE" "${JSON_PATH:-}"

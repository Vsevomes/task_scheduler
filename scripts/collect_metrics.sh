#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-1}"
OUT="${2:-results/system_metrics.csv}"

mkdir -p "$(dirname "$OUT")"
echo "timestamp,cpu_user,cpu_sys,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb,mem_used_mb,mem_total_mb" >"$OUT"

while true; do
  TS="$(date +%s)"
  CPU="$(mpstat 1 1 2>/dev/null | awk '/Average/ {printf "%.2f,%.2f", $3, $5}' || echo "0,0")"
  GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0,0,0")"
  MEM="$(free -m | awk '/Mem:/ {print $3 "," $2}')"
  IFS=',' read -r GPU_UTIL GPU_MEM GPU_TOTAL <<<"$GPU"
  IFS=',' read -r MEM_USED MEM_TOTAL <<<"$MEM"
  echo "$TS,$CPU,$GPU_UTIL,$GPU_MEM,$GPU_TOTAL,$MEM_USED,$MEM_TOTAL" >>"$OUT"
  sleep "$INTERVAL"
done

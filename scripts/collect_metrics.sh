#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-1}"
OUT="${2:-results/system_metrics.csv}"
CORES_OUT="${OUT%.csv}_cores.csv"

# Sample window inside each interval (seconds).
SAMPLE_SEC="${METRICS_SAMPLE_SEC:-0.4}"

mkdir -p "$(dirname "$OUT")"
echo "timestamp,cpu_user,cpu_sys,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb,mem_used_mb,mem_total_mb" >"$OUT"
echo "timestamp,core,usr,sys,idle,used" >"$CORES_OUT"

read_proc_stat() {
  awk '/^cpu[0-9]+/ {
    core = substr($1, 4)
    idle = $5
    total = 0
    for (i = 2; i <= NF; i++)
      total += $i
    print core, total, idle, $2, $4
  }' /proc/stat
}

while true; do
  TS="$(date +%s)"
  mapfile -t STAT1 < <(read_proc_stat)

  GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0,0,0")"
  MEM="$(free -m | awk '/Mem:/ {print $3 "," $2}')"
  IFS=',' read -r GPU_UTIL GPU_MEM GPU_TOTAL <<<"$GPU"
  IFS=',' read -r MEM_USED MEM_TOTAL <<<"$MEM"

  sleep "$SAMPLE_SEC"

  mapfile -t STAT2 < <(read_proc_stat)

  TOTAL_DELTA=0
  IDLE_DELTA=0
  USR_SUM=0
  SYS_SUM=0
  for ((i = 0; i < ${#STAT1[@]}; i++)); do
    read -r CORE T1 I1 U1 S1 <<<"${STAT1[$i]}"
    read -r _ T2 I2 U2 S2 <<<"${STAT2[$i]}"
    DT=$((T2 - T1))
    DI=$((I2 - I1))
    DU=$((U2 - U1))
    DS=$((S2 - S1))
    if ((DT > 0)); then
      USED=$(awk -v dt="$DT" -v di="$DI" 'BEGIN {printf "%.2f", 100.0 * (dt - di) / dt}')
      USR=$(awk -v dt="$DT" -v du="$DU" 'BEGIN {printf "%.2f", 100.0 * du / dt}')
      SYS=$(awk -v dt="$DT" -v ds="$DS" 'BEGIN {printf "%.2f", 100.0 * ds / dt}')
      IDLE=$(awk -v dt="$DT" -v di="$DI" 'BEGIN {printf "%.2f", 100.0 * di / dt}')
      echo "$TS,$CORE,$USR,$SYS,$IDLE,$USED" >>"$CORES_OUT"
      TOTAL_DELTA=$((TOTAL_DELTA + DT))
      IDLE_DELTA=$((IDLE_DELTA + DI))
      USR_SUM=$((USR_SUM + DU))
      SYS_SUM=$((SYS_SUM + DS))
    fi
  done

  if ((TOTAL_DELTA > 0)); then
    CPU_USER=$(awk -v u="$USR_SUM" -v t="$TOTAL_DELTA" 'BEGIN {printf "%.2f", 100.0 * u / t}')
    CPU_SYS=$(awk -v s="$SYS_SUM" -v t="$TOTAL_DELTA" 'BEGIN {printf "%.2f", 100.0 * s / t}')
  else
    CPU_USER="0.00"
    CPU_SYS="0.00"
  fi

  echo "$TS,$CPU_USER,$CPU_SYS,$GPU_UTIL,$GPU_MEM,$GPU_TOTAL,$MEM_USED,$MEM_TOTAL" >>"$OUT"

  REMAIN="$(python3 -c "print(max(0.0, float('$INTERVAL') - float('$SAMPLE_SEC')))")"
  if python3 -c "exit(0 if float('$REMAIN') > 0 else 1)"; then
    sleep "$REMAIN"
  fi
done

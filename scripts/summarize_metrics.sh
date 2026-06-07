#!/usr/bin/env bash
# Print memory, CPU/GPU utilization, active cores, and benchmark overhead summary.
set -euo pipefail

MAIN_CSV="${1:-}"
JSON_PATH="${2:-}"

if [[ -z "$MAIN_CSV" || ! -f "$MAIN_CSV" ]]; then
  echo "Usage: $0 system_metrics.csv [result.json]"
  exit 1
fi

CORES_CSV="${MAIN_CSV%.csv}_cores.csv"
HOT_CORE_AVG="${METRICS_HOT_CORE_AVG:-5.0}"
HOT_CORE_MAX="${METRICS_HOT_CORE_MAX:-20.0}"

python3 << PY
import csv
import json
import os

main_csv = "$MAIN_CSV"
cores_csv = "$CORES_CSV"
json_path = "$JSON_PATH"
hot_avg = float("$HOT_CORE_AVG")
hot_max = float("$HOT_CORE_MAX")

def read_main():
    with open(main_csv, newline="") as f:
        return list(csv.DictReader(f))

def read_cores():
    if not os.path.exists(cores_csv):
        return []
    with open(cores_csv, newline="") as f:
        return list(csv.DictReader(f))

def read_json():
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)

def fvals(rows, key):
    out = []
    for row in rows:
        if row.get(key) not in (None, ""):
            try:
                out.append(float(row[key]))
            except ValueError:
                pass
    return out

main = read_main()
cores = read_cores()
result = read_json()

print("=== Run summary ===")
if result:
    print("Scenario: %s  mode: %s" % (result.get("scenario", "?"), result.get("mode", "?")))
    params = result.get("params", {})
    if params:
        print("Params: %s" % ", ".join("%s=%s" % (k, v) for k, v in sorted(params.items())))

print()
print("--- Memory ---")
if main:
    mem_used = fvals(main, "mem_used_mb")
    mem_total = fvals(main, "mem_total_mb")
    gpu_mem = fvals(main, "gpu_mem_used_mb")
    gpu_total = fvals(main, "gpu_mem_total_mb")
    if mem_used:
        print("RAM:  used avg=%.0f MB  max=%.0f MB  of %.0f MB (%.1f%% peak)" % (
            sum(mem_used) / len(mem_used), max(mem_used),
            mem_total[-1] if mem_total else 0,
            100.0 * max(mem_used) / mem_total[-1] if mem_total and mem_total[-1] else 0))
    if gpu_mem:
        print("VRAM: used avg=%.0f MB  max=%.0f MB  of %.0f MB (%.1f%% peak)" % (
            sum(gpu_mem) / len(gpu_mem), max(gpu_mem),
            gpu_total[-1] if gpu_total else 0,
            100.0 * max(gpu_mem) / gpu_total[-1] if gpu_total and gpu_total[-1] else 0))
else:
    print("No memory samples")

print()
print("--- Utilization ---")
if main:
    gpu = fvals(main, "gpu_util")
    cpu_user = fvals(main, "cpu_user")
    cpu_sys = fvals(main, "cpu_sys")
    if gpu:
        print("GPU compute util: avg=%.1f%%  max=%.1f%%  samples=%d" % (
            sum(gpu) / len(gpu), max(gpu), len(gpu)))
    if cpu_user:
        print("CPU total: usr avg=%.1f%% max=%.1f%% | sys avg=%.1f%% max=%.1f%% | samples=%d" % (
            sum(cpu_user) / len(cpu_user), max(cpu_user),
            sum(cpu_sys) / len(cpu_sys) if cpu_sys else 0.0,
            max(cpu_sys) if cpu_sys else 0.0,
            len(cpu_user)))

if cores:
    by_core = {}
    for row in cores:
        cid = row["core"]
        used = float(row["used"])
        usr = float(row.get("usr") or 0)
        sys_ = float(row.get("sys") or 0)
        by_core.setdefault(cid, {"used": [], "usr": [], "sys": []})
        by_core[cid]["used"].append(used)
        by_core[cid]["usr"].append(usr)
        by_core[cid]["sys"].append(sys_)

    ranked = []
    for cid, vals in by_core.items():
        avg_used = sum(vals["used"]) / len(vals["used"])
        peak_used = max(vals["used"])
        avg_usr = sum(vals["usr"]) / len(vals["usr"])
        avg_sys = sum(vals["sys"]) / len(vals["sys"])
        ranked.append((peak_used, avg_used, cid, avg_usr, avg_sys, len(vals["used"])))
    ranked.sort(reverse=True)

    hot = [r for r in ranked if r[1] >= hot_avg or r[0] >= hot_max]
    idle = [r for r in ranked if r[1] < 1.0 and r[0] < 5.0]

    print()
    print("--- CPU cores ---")
    print("Per-core load (sorted by peak used %%):")
    for peak, avg, cid, avg_usr, avg_sys, n in ranked:
        tag = ""
        if avg >= hot_avg or peak >= hot_max:
            tag = "  <- active"
        print("  core %2s: avg=%5.1f%%  peak=%5.1f%%  (usr=%.1f sys=%.1f) samples=%d%s" % (
            cid, avg, peak, avg_usr, avg_sys, n, tag))

    if hot:
        print()
        print("Active cores (avg>=%.0f%% or peak>=%.0f%%): %s" % (
            hot_avg, hot_max, ", ".join("CPU%s" % r[2] for r in hot)))
    if idle:
        print("Mostly idle: %s" % ", ".join("CPU%s" % r[2] for r in idle[:8]))
        if len(idle) > 8:
            print("  ... and %d more idle cores" % (len(idle) - 8))

    all_used = [float(r["used"]) for r in cores]
    print("All cores: avg used=%.1f%%  peak on one core=%.1f%%" % (
        sum(all_used) / len(all_used), max(all_used)))
else:
    print("No per-core samples (%s)" % cores_csv)

print()
print("--- Timing / overhead ---")
if result:
    m = result.get("metrics", {})
    total = m.get("total_time_ms")
    if total is not None:
        print("Total wall time: %.1f ms" % total)

    starpu_keys = [
        ("starpu_init_ms", "StarPU init"),
        ("starpu_data_registration_ms", "Data registration"),
        ("starpu_task_submission_ms", "Task submission"),
        ("starpu_wait_ms", "Task wait (compute+transfer)"),
        ("starpu_data_unregister_ms", "Data unregister"),
        ("starpu_data_unregister_copyback_ms", "Unregister + copyback"),
    ]
    phases = [(label, m[key]) for key, label in starpu_keys if key in m]
    if phases:
        overhead = sum(v for _, v in phases if _ != "Task wait (compute+transfer)")
        compute = next((v for lbl, v in phases if lbl == "Task wait (compute+transfer)"), 0.0)
        print("StarPU breakdown:")
        for label, val in phases:
            pct = 100.0 * val / total if total else 0
            print("  %-28s %8.1f ms  (%4.1f%%)" % (label, val, pct))
        print("  %-28s %8.1f ms  (%4.1f%%)" % (
            "Overhead (excl. wait)", overhead, 100.0 * overhead / total if total else 0))
        print("  %-28s %8.1f ms  (%4.1f%%)" % (
            "Useful wait phase", compute, 100.0 * compute / total if total else 0))
    elif total is not None:
        print("Native path: no StarPU phase breakdown (total = compute + I/O)")

    if m.get("gflops"):
        print("Performance: %.2f GFLOPS" % m["gflops"])
    if m.get("task_count"):
        print("Tasks/tiles: %d" % int(m["task_count"]))
else:
    print("No result JSON (pass path as 2nd argument for overhead breakdown)")

print()
print("--- Metrics files ---")
print(" ", main_csv)
if os.path.exists(cores_csv):
    print(" ", cores_csv)
if json_path and os.path.exists(json_path):
    print(" ", json_path)
PY

#!/usr/bin/env python3
"""Aggregate experiment_results JSON and print markdown tables."""
import json
import glob
import os
import statistics
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "experiment_results")


def load_records():
    records = []
    for pattern in ("*/*/*.json", "*/*/*/*.json"):
        for path in glob.glob(os.path.join(ROOT, pattern)):
            with open(path) as f:
                d = json.load(f)
            d["_ts"] = os.path.basename(path).replace(".json", "")
            d["_path"] = path.replace(ROOT + os.sep, "")
            records.append(d)
    return records


def group_latest(records, scenario=None):
    groups = defaultdict(list)
    for r in records:
        if scenario and r["scenario"] != scenario:
            continue
        params = tuple(sorted((k, str(v)) for k, v in r.get("params", {}).items()))
        groups[(r["scenario"], r["mode"], params)].append(r)
    out = {}
    for key, items in groups.items():
        items.sort(key=lambda x: x["_ts"])
        times = [x["metrics"].get("total_time_ms", 0) for x in items]
        out[key] = {
            "latest": items[-1],
            "count": len(items),
            "median_ms": statistics.median(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        }
    return out


def params_dict(key):
    return dict(key[2])


def fmt_ms(v):
    return f"{v:.1f}" if v is not None else "—"


def fmt_gflops(v):
    return f"{v:.1f}" if v is not None else "—"


if __name__ == "__main__":
    recs = load_records()
    print(f"Loaded {len(recs)} JSON files from {ROOT}")

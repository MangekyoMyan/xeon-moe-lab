#!/usr/bin/env python3
"""Summarize phase captures without treating sampled cycles as wall-clock fractions."""
import argparse
import csv
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("capture", type=Path)
parser.add_argument("flat_report", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

def counters(path):
    result = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        row = next(csv.reader([line]))
        if len(row) < 7:
            raise ValueError("Unexpected perf CSV row: " + line)
        result.append({"value": float(row[0]), "unit": row[1],
                       "event": ",".join(row[2:-4]), "runtime_ns": float(row[-4]),
                       "percent_running": float(row[-3])})
    if not result or any(r["percent_running"] < 99.9 for r in result):
        raise ValueError("Missing or multiplexed counters require explicit review")
    return result

capture = json.loads((args.capture / "capture.json").read_text())
if not capture["valid_sample"]:
    raise ValueError("Invalid profile window")
core, imc = counters(args.capture / "core.csv"), counters(args.capture / "imc.csv")

def value(rows, event):
    return sum(r["value"] for r in rows if r["event"] == event)

core_seconds = value(core, "duration_time") / 1e9
imc_seconds = value(imc, "duration_time") / 1e9
if core_seconds <= 0 or imc_seconds <= 0:
    raise ValueError("Explicit duration_time is required")
symbols = []
flat = args.flat_report.read_text()
for line in flat.splitlines():
    match = re.match(r"\s*([\d.]+)%\s+(\S+)\s+\[[^\]]+\]\s+(.*)", line)
    if match:
        symbols.append({"percent_sampled_cycles": float(match[1]), "dso": match[2], "symbol": match[3]})
if not symbols or abs(sum(r["percent_sampled_cycles"] for r in symbols) - 100) > 2:
    raise ValueError("Flat symbol report could not be fully parsed")
lost = re.search(r"Total Lost Samples:\s*(\d+)", flat)
report = {
    "capture": str(args.capture), "flat_report": str(args.flat_report),
    "phase": capture["arguments"]["phase"], "boot_id": capture["boot_id"],
    "valid_approximate_sample": True, "nominal_margin_seconds": capture["nominal_margin_seconds"],
    "sampled_cycles_percent_note": "Includes memory stalls. OpenMP percentage is sampled CPU cycles, not elapsed-time fraction.",
    "lost_samples": int(lost[1]) if lost else None,
    "ipc": value(core, "instructions:u") / value(core, "cycles:u"),
    "cycles_stalled_with_l2_miss_pending_fraction": value(core, "cycle_activity.stalls_l2_pending:u") / value(core, "cycles:u"),
    "stall_counter_note": "Execution stalls with an outstanding L2-miss load; this is not an exclusive DRAM-bandwidth-bound fraction and must not be added to sampled function percentages.",
    "cpu_utilization_equivalent": value(core, "task-clock") / (core_seconds * 1000),
    "core_seconds": core_seconds, "imc_seconds": imc_seconds,
    "dram_read_GB_per_second": sum(r["value"] for r in imc if "umask=0x03" in r["event"]) * 64 / imc_seconds / 1e9,
    "dram_write_GB_per_second": sum(r["value"] for r in imc if "umask=0x0c" in r["event"]) * 64 / imc_seconds / 1e9,
    "memory_counter_note": "System-wide sum across IMCs, 64 bytes per CAS; duration_time is not multiplied or divided by the number of counters.",
    "cpu_migrations": value(core, "cpu-migrations"), "minor_faults": value(core, "minor-faults"),
    "major_faults": value(core, "major-faults"), "context_switches": value(core, "context-switches"),
    "window_disk_read_bytes": int(capture["window_end_snapshot"]["io"]["read_bytes"]) - int(capture["window_start_snapshot"]["io"]["read_bytes"]),
    "openmp_percent_sampled_cycles": sum(r["percent_sampled_cycles"] for r in symbols if "libgomp" in r["dso"]),
    "q8_dot_percent_sampled_cycles": sum(r["percent_sampled_cycles"] for r in symbols if "ggml_vec_dot_q8_0_q8_0" in r["symbol"]),
    "iq2_dot_percent_sampled_cycles": sum(r["percent_sampled_cycles"] for r in symbols if "ggml_vec_dot_iq2_xxs_q8_K" in r["symbol"]),
    "symbols": symbols, "core_counters": core, "imc_counters": imc,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("symbols", "core_counters", "imc_counters")}, indent=2))

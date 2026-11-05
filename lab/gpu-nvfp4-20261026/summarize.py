#!/usr/bin/env python3
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
results = root.parent / "results" / root.name
summary = {}
for folder in sorted(results.iterdir()):
    if not (folder / "run.json").exists():
        continue
    run = json.loads((folder / "run.json").read_text())
    rows = run.get("samples", [])
    entry = {k: run.get(k) for k in ["returncode", "error", "elapsed_s", "command", "extra_env", "fan_duty_before", "fan_duty_after"]}
    if rows:
        entry["maxima"] = {k: max(s["gpu"][k] for s in rows if s["gpu"][k] is not None) for k in
            ["gpu_C", "hbm_C", "power_W", "vram_used_MiB", "pcie_rx_KB_s", "pcie_tx_KB_s"]}
        entry["maxima"]["cpu_C"] = max(max(s["cpu_C"].values(), default=0) for s in rows)
        entry["maxima"]["fan_PWM_percent"] = max(max(s["fan_duty"]["percent"]) for s in rows if "fan_duty" in s)
        active = [s["gpu"] for s in rows if s["gpu"]["gpu_util"] >= 90]
        entry["active_sample_count"] = len(active)
        if active:
            entry["active_sample_medians"] = {k: statistics.median(s[k] for s in active) for k in
                ["power_W", "sm_MHz", "mem_MHz", "memory_util", "pcie_rx_KB_s", "pcie_tx_KB_s"]}
            entry["active_power_cap_sample_fraction"] = sum(s["throttling"].get("clocks_event_reason_sw_power_cap") == "Active" for s in active) / len(active)
    stdout = (folder / "stdout.txt").read_text()
    try:
        entry["bench"] = [{k: b[k] for k in ["n_prompt", "n_gen", "n_batch", "n_ubatch", "n_threads", "avg_ts", "stddev_ts", "samples_ts"]} for b in json.loads(stdout)]
    except (ValueError, TypeError, KeyError):
        pass
    summary[folder.name] = entry

(root / "results-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for label, entry in summary.items():
    if "bench" in entry:
        print(label, [(b["n_prompt"], b["n_gen"], round(b["avg_ts"], 3)) for b in entry["bench"]], entry.get("maxima"))

text = (results / "quant-kernel-perf/stdout.txt").read_text()
kernel = []
for match in re.finditer(r"type_a=([^,]+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),[^\n]*?([\d.]+) us/run", text):
    dtype, m, n, k, us = match.groups()
    row = {"type": dtype, "m": int(m), "n": int(n), "k": int(k), "us": float(us)}
    bits = {"nvfp4": 4.5, "mxfp4": 4.25, "q8_0": 8.5, "q4_K": 4.5}[dtype]
    row["weight_bytes_per_elapsed_GB_s_proxy"] = row["m"] * row["k"] * bits / 8 / (row["us"] * 1e3)
    row["equivalent_TFLOPS"] = 2 * row["m"] * row["n"] * row["k"] / (row["us"] * 1e6)
    kernel.append(row)
(root / "kernel-summary.json").write_text(json.dumps(kernel, indent=2) + "\n")

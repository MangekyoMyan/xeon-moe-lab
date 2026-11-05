#!/usr/bin/env python3
"""Summarize completed model runs, retaining individual timed samples."""
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
rows = []
for folder in sorted(OUT.glob("model-*/")):
    if not (folder / "run.json").exists():
        continue
    record = json.loads((folder / "run.json").read_text())
    if record.get("returncode") != 0 or "error" in record:
        continue
    try:
        benchmarks = json.loads((folder / "stdout.txt").read_text())
    except json.JSONDecodeError:
        continue
    for benchmark in benchmarks:
        mode = "cpu" if benchmark["n_gpu_layers"] == 0 else "common" if benchmark["n_cpu_moe"] == 46 else "experts10"
        task = "prompt" if benchmark["n_prompt"] else "generation"
        phases = record.get("phases", [])
        selected = []
        for sample in record["samples"]:
            prior = [p for p in phases if p["elapsed_s"] <= sample["elapsed_s"]]
            if prior and re.search(r": " + task + r" run \d+/\d+$", prior[-1]["message"]):
                selected.append(sample)
        row = {"label": folder.name, "mode": mode, "task": task,
               "threads": benchmark["n_threads"], "prompt_tokens": benchmark["n_prompt"],
               "generation_tokens": benchmark["n_gen"], "depth": benchmark["n_depth"],
               "poll": benchmark["poll"], "boot_id": record["boot_id"],
               "no_op_offload": benchmark["no_op_offload"],
               "graphs": "off" if "GGML_CUDA_DISABLE_GRAPHS=1" in record.get("extra_env", []) else "on",
               "samples_ts": benchmark["samples_ts"], "samples_ns": benchmark["samples_ns"],
               "mean_ts": benchmark["avg_ts"], "stddev_ts": benchmark["stddev_ts"],
               "max_process_disk_read_bytes": max((s["process"]["disk_read_bytes"] for s in record["samples"] if s.get("process")), default=0),
               "max_vram_MiB": max(s["gpu"]["vram_used_MiB"] for s in record["samples"]),
               "max_gpu_C": max(s["gpu"]["gpu_C"] for s in record["samples"]),
               "max_hbm_C": max(s["gpu"]["hbm_C"] for s in record["samples"]),
               "telemetry_samples_in_phase": len(selected)}
        if selected:
            row["phase_gpu_W_mean"] = statistics.mean(s["gpu"]["power_W"] for s in selected)
            row["phase_gpu_util_mean_percent"] = statistics.mean(s["gpu"]["gpu_util"] for s in selected)
            row["phase_sm_MHz_mean"] = statistics.mean(s["gpu"]["sm_MHz"] for s in selected)
            row["phase_memory_MHz_min"] = min(s["gpu"]["mem_MHz"] for s in selected)
        if len(selected) >= 2 and selected[0].get("host_energy"):
            first, last = selected[0], selected[-1]
            seconds = last["elapsed_s"] - first["elapsed_s"]
            joules = sum((last["host_energy"][k]["uj"] - v["uj"]) % v["range"] / 1e6
                         for k, v in first["host_energy"].items())
            row["phase_host_cpu_plus_dram_RAPL_W"] = joules / seconds
        fan_max = {}
        for bmc in record.get("bmc", []):
            for line in bmc["sensors"].splitlines():
                fields = [s.strip() for s in line.split("|")]
                if len(fields) >= 3 and fields[2] == "RPM" and fields[1] != "na":
                    fan_max[fields[0]] = max(fan_max.get(fields[0], 0), float(fields[1]))
        row["fan_rpm_max"] = fan_max
        rows.append(row)

groups = {}
for row in rows:
    key = (row["mode"], row["task"], row["threads"], row["prompt_tokens"], row["generation_tokens"], row["depth"], row["poll"], row["graphs"], row["boot_id"], row["no_op_offload"])
    groups.setdefault(key, []).append(row)
summary = []
for key, group in groups.items():
    values = [v for row in group for v in row["samples_ts"]]
    summary.append({"mode": key[0], "task": key[1], "threads": key[2], "prompt_tokens": key[3],
                    "generation_tokens": key[4], "depth": key[5], "poll": key[6], "graphs": key[7], "boot_id": key[8],
                    "no_op_offload": key[9],
                    "sample_count": len(values),
                    "mean_ts": statistics.mean(values), "median_ts": statistics.median(values),
                    "min_ts": min(values), "max_ts": max(values),
                    "stddev_ts": statistics.stdev(values) if len(values) > 1 else 0,
                    "labels": [r["label"] for r in group], "all_timing_samples": values,
                    "max_disk_read_bytes": max(r["max_process_disk_read_bytes"] for r in group),
                    "max_vram_MiB": max(r["max_vram_MiB"] for r in group),
                    "max_gpu_C": max(r["max_gpu_C"] for r in group),
                    "max_hbm_C": max(r["max_hbm_C"] for r in group)})
report = {"completed_rows": rows, "summary": summary,
          "scope": "llama-bench random-token prompt/generation tests. One process per placement; repeated in reverse order.",
          "telemetry_note": "GPU power/utilization sampled at about 1 s. Phase boundaries are read from progress messages, also at about 1 s."}
(OUT / "model-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(summary, indent=2))

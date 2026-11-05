#!/usr/bin/env python3
"""Measure identical greedy Japanese replies in a fresh-process BAAB sequence with light telemetry."""
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

lab = Path(__file__).resolve().parent
output = lab / "results/optimization-memory-sync-20261019/response-reverse.json"
if output.exists():
    raise SystemExit("Preserve the existing response comparison")
binaries = {"A": lab / "build-avx-deep/bin/llama-completion", "B": lab / "build-avx-prefetch/bin/llama-completion"}
env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24", OMP_MAX_ACTIVE_LEVELS="1",
           GGML_CPU_NUMA_REPLICA="0", GGML_CPU_MOE_PAIR_TRACE="0", GGML_CPU_NUMA_REPLICA_TRACE="0")
runs = []
for index, variant in enumerate("BAAB"):
    label = f"prefetch-response-baab-{index}-{variant}"
    print("starting " + label, flush=True)
    command = [sys.executable, str(lab / "verify_deepseek_light.py"), "--binary", str(binaries[variant]),
               "--label", label, "--threads", "12", "--threads-batch", "24", "--batch", "512", "--ubatch", "512",
               "--note-repetitions", "0"]
    with (lab / "results" / (label + ".log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    paths = list((lab / "results").glob("deepseek-functional-" + label + "-*/summary.json"))
    if len(paths) != 1:
        raise RuntimeError("Expected one fresh response result")
    result = json.loads(paths[0].read_text())
    count = re.search(r"common_perf_print:\s+eval time\s*=.*?/\s*(\d+) runs", "\n".join(result["timing_lines"]))
    if not count:
        raise RuntimeError("Missing evaluated-token count")
    runs.append({"variant": variant, "summary": str(paths[0]), "response": result["response"],
                 "boot_id": result["boot_id"], "timings_ms": result["timings_ms"],
                 "evaluated_tokens": int(count[1]),
                 "timing_lines": result["timing_lines"],
                 "max_observed_disk_read_bytes": max(r.get("disk_read_bytes", 0) for r in result["observations"])})
    output.write_text(json.dumps({"runs": runs, "complete": False}, ensure_ascii=False, indent=2) + "\n")
medians = {variant: {phase: statistics.median(r["timings_ms"][phase] for r in runs if r["variant"] == variant)
                     for phase in ("load", "prompt eval", "eval")} for variant in "AB"}
report = {"runs": runs, "complete": True, "responses_identical": len({r["response"] for r in runs}) == 1,
          "evaluated_token_counts_identical": len({r["evaluated_tokens"] for r in runs}) == 1,
          "same_boot": len({r["boot_id"] for r in runs}) == 1,
          "warm_cache_observed": all(r["max_observed_disk_read_bytes"] == 0 for r in runs),
          "median_ms": medians,
          "generation_throughput_gain_percent": 100 * (medians["A"]["eval"] / medians["B"]["eval"] - 1),
          "scope": "One fixed 59-token Japanese prompt and identical greedy response. Compare evaluated-token counts in timing_lines; no broad workload claim."}
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["responses_identical"] or not report["evaluated_token_counts_identical"] or not report["same_boot"] or not report["warm_cache_observed"]:
    raise SystemExit("Review response comparison conditions")

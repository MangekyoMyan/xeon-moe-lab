#!/usr/bin/env python3
"""Capture baseline/final CPU and DRAM profiles sequentially on an idle host."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
results = lab / "results"
baseline = lab / "build-avx-unpack/bin/llama-bench"
final = lab / "build-avx-deep/bin/llama-bench"
records = []
for key, phase, bench in (("baseline-decode", "decode", baseline),
                           ("final-decode", "decode", final), ("final-prefill", "prefill", final)):
    output = results / ("deep-profile-" + key)
    command = ["sudo", "-n", "env", "OMP_DYNAMIC=FALSE", "OMP_THREAD_LIMIT=24", "OMP_MAX_ACTIVE_LEVELS=1",
        "GGML_CPU_NUMA_REPLICA=0", "GGML_CPU_MOE_PAIR_TRACE=0", "GGML_CPU_NUMA_REPLICA_TRACE=0",
        sys.executable, str(lab / "capture_deep_profile.py"), phase, "--bench", str(bench), "--output", str(output)]
    print("starting " + key, flush=True)
    with (results / ("deep-review-profile-" + key + ".log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    flat = results / ("deep-review-profile-" + key + ".txt")
    with flat.open("w") as log:
        subprocess.run(["sudo", "-n", "perf", "report", "--stdio", "--no-children", "--sort", "dso,symbol",
                        "--percent-limit", "0", "-i", str(output / "profile.data")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = results / ("deep-review-profile-" + key + ".json")
    subprocess.run([sys.executable, str(lab / "summarize_deep_profiles.py"), str(output), str(flat),
                    "--output", str(summary)], check=True)
    records.append({"key": key, "command": command, "summary": str(summary)})
    (results / "deep-review-profile-index.json").write_text(json.dumps(records, indent=2) + "\n")
    print("finished " + key, flush=True)

#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["decode", "prefill"])
args = parser.parse_args()
lab = Path(__file__).resolve().parent
baseline = lab / "review-deep-pools-20261019/baseline/build/bin/llama-bench"
combined = lab / "review-deep-combined-20261019/build/bin/llama-bench"
pthread = lab / "review-deep-pools-20261019/pthread/build/bin/llama-bench"
if args.phase == "decode":
    jobs = [
        ("combined-abba", baseline, combined, "ABBA", "0", "0", "Combined MoE+Q8, no extra weight memory"),
        ("combined-baab", baseline, combined, "BAAB", "0", "0", "Combined MoE+Q8, no extra weight memory, reverse confirmation"),
        ("combined-numa-abba", combined, combined, "ABBA", "0", "1", "Incremental NUMA replica benefit on combined build"),
    ]
    shape = ["--prompt", "0", "--tokens", "32", "--threads", "12", "--repetitions", "3"]
else:
    jobs = [
        ("pools-prefill", baseline, pthread, "ABBA", "0", "0", "OpenMP ON versus OFF, matched 512-token prefill, 24 workers"),
        ("combined-prefill", baseline, combined, "ABBA", "0", "0", "Combined MoE+Q8 prefill regression check, 512 tokens, 24 workers"),
    ]
    shape = ["--prompt", "512", "--tokens", "0", "--threads", "24", "--repetitions", "1"]
for label, a, b, order, numa_a, numa_b, note in jobs:
    command = [sys.executable, str(lab / "run_cpu_ab.py"), "--lab", str(lab),
        "--baseline", str(a), "--candidate", str(b), "--order", order,
        "--baseline-env", "GGML_CPU_NUMA_REPLICA=" + numa_a,
        "--candidate-env", "GGML_CPU_NUMA_REPLICA=" + numa_b,
        "--depth", "0", "--note", note, *shape]
    print("starting " + label, flush=True)
    with (lab / "results" / ("deep-review-" + label + ".log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("finished " + label, flush=True)

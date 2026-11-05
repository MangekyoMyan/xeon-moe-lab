#!/usr/bin/env python3
"""Measure reduced thread polling/physical threads after placement validation."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / "results" / ROOT.name
started = time.monotonic()
while True:
    assert not (ROOT / "STOP_REQUESTED").exists(), "Experiments stopped"
    try:
        cases = json.loads((OUT / "fixed-offload-sequence-summary.json").read_text())
        if len(cases) == 4:
            break
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    assert time.monotonic() - started < 1200
    time.sleep(5)
time.sleep(15)
scores = {}
for case in cases:
    for benchmark in case["benchmarks"]:
        if benchmark["n_gen"]:
            scores.setdefault(case["mode"], []).extend(benchmark["samples_ts"])
mode = max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
results = []
for threads in [24, 12]:
    assert not (ROOT / "STOP_REQUESTED").exists()
    label = f"model-quiet-t{threads}-poll0"
    print("Starting", label, mode, flush=True)
    with (OUT / (label + "-console.log")).open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "model_case.py"), "--label", label, "--mode", mode,
                        "--prompt", "64", "--tokens", "64", "--repetitions", "2", "--quiet",
                        "--op-offload", "off", "--poll", "0", "--threads", str(threads)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    benchmarks = json.loads((OUT / label / "stdout.txt").read_text())
    results.append({"label": label, "mode": mode, "threads": threads, "benchmarks": benchmarks})
    (OUT / "quiet-tuning-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Completed", label, [(b["n_prompt"], b["n_gen"], b["avg_ts"]) for b in benchmarks], flush=True)
    time.sleep(15)

assert not (ROOT / "STOP_REQUESTED").exists()
label = "accepted-cpu-night-reference"
model = LAB / "models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf"
command = [sys.executable, str(ROOT / "watch.py"), "--label", label, "--timeout", "300", "--quiet", "--",
           sys.executable, str(LAB / "without_thp.py"), "numactl", "--physcpubind=0-23", "--interleave=0,1",
           str(LAB / "review-glm-q8-columns4-20261020/build/bin/llama-bench"), "-m", str(model),
           "-t", "24", "-p", "64", "-n", "64", "-r", "2", "-b", "128", "-ub", "128", "--poll", "50",
           "-ngl", "0", "-dev", "none", "-nkvo", "1", "-nopo", "1", "-lm", "mmap", "-lzm", "off", "--repack", "0",
           "--cpu-mask", "0xffffff", "--cpu-strict", "1", "--offline", "--progress", "-o", "json"]
print("Starting", label, flush=True)
with (OUT / (label + "-console.log")).open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
print("Completed", label, flush=True)

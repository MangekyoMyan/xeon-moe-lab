#!/usr/bin/env python3
"""Run single-core kernel comparisons sequentially on the actual Xeon."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

lab = Path(__file__).resolve().parent
binary = lab / "review-deep-micro-20261019/kernel-bench"
out = lab / "results" / ("deep-kernel-micro-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
out.mkdir()
report = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "runs": []}
for kind, k, m, iterations in [("q8", 1024, 32768, 96), ("q8", 8192, 4096, 96),
                               ("iq2", 4096, 32, 16384), ("iq2", 4096, 2048, 256)]:
    for order in ("forward", "reverse"):
        name = f"{kind}-{k}-{m}-{order}"
        command = ["numactl", "--physcpubind=0", "--membind=0", str(binary),
                   kind, str(k), str(m), str(iterations), "6", order]
        start = time.monotonic()
        print(json.dumps({"starting": name, "command": command}), flush=True)
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        (out / (name + ".stderr.txt")).write_text(proc.stderr)
        (out / (name + ".json")).write_text(proc.stdout)
        proc.check_returncode()
        data = json.loads(proc.stdout)
        medians = {key: statistics.median(values) for key, values in data["seconds"].items()}
        gains = {key: 100 * (medians["baseline"] / value - 1) for key, value in medians.items()}
        row = {"name": name, "command": command, "wall_seconds": time.monotonic() - start,
               "median_seconds": medians, "gain_percent": gains, "raw": data}
        report["runs"].append(row)
        (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"finished": name, "gain_percent": gains}), flush=True)
print(json.dumps({"saved": str(out / "comparison.json")}), flush=True)

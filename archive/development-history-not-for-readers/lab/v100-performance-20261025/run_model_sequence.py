#!/usr/bin/env python3
"""Wait for the cold load, then measure placements in A-B-C-C-B-A order."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
warm = OUT / "cold-model-warmup/run.json"
started = time.monotonic()
while True:
    if (ROOT / "STOP_REQUESTED").exists():
        raise SystemExit("Stopped before starting model experiments")
    try:
        report = json.loads(warm.read_text())
        if "returncode" in report:
            assert report["returncode"] == 0 and "error" not in report, "Cold load failed"
            break
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    assert time.monotonic() - started < 3600, "Cold load did not finish"
    time.sleep(5)

sequence = [("model-a1-cpu", "cpu"), ("model-b1-common", "common"),
            ("model-c1-experts10", "experts10"), ("model-c2-experts10", "experts10"),
            ("model-b2-common", "common"), ("model-a2-cpu", "cpu")]
results = []
for label, mode in sequence:
    if (ROOT / "STOP_REQUESTED").exists():
        raise SystemExit("Stopped at a completed model case")
    print("Starting", label, flush=True)
    with (OUT / (label + "-console.log")).open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "model_case.py"), "--label", label, "--mode", mode],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    benchmarks = json.loads((OUT / label / "stdout.txt").read_text())
    results.append({"label": label, "mode": mode, "benchmarks": benchmarks})
    (OUT / "model-sequence-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Completed", label, [(b["n_prompt"], b["n_gen"], b["avg_ts"]) for b in benchmarks], flush=True)
print("Model sequence complete", flush=True)

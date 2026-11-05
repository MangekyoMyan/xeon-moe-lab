#!/usr/bin/env python3
"""Compare explicit CPU-expert execution after the default placement screen."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
started = time.monotonic()
while True:
    assert not (ROOT / "STOP_REQUESTED").exists(), "Experiments stopped"
    try:
        cases = json.loads((OUT / "night-model-sequence-summary.json").read_text())
        if len(cases) == 6:
            break
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    assert time.monotonic() - started < 1800, "Previous sequence did not complete"
    time.sleep(5)
time.sleep(15)
results = []
for label, mode in [("model-fixed-b1-common", "common"), ("model-fixed-c1-experts10", "experts10"),
                    ("model-fixed-c2-experts10", "experts10"), ("model-fixed-b2-common", "common")]:
    assert not (ROOT / "STOP_REQUESTED").exists(), "Experiments stopped"
    print("Starting", label, flush=True)
    with (OUT / (label + "-console.log")).open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "model_case.py"), "--label", label, "--mode", mode,
                        "--prompt", "64", "--tokens", "64", "--repetitions", "2", "--quiet", "--op-offload", "off"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    benchmarks = json.loads((OUT / label / "stdout.txt").read_text())
    results.append({"label": label, "mode": mode, "benchmarks": benchmarks})
    (OUT / "fixed-offload-sequence-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Completed", label, [(b["n_prompt"], b["n_gen"], b["avg_ts"]) for b in benchmarks], flush=True)
    time.sleep(15)

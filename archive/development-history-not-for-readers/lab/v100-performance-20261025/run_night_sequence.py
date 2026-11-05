#!/usr/bin/env python3
"""Resume the real-model comparison after NIC removal, with quiet-session limits."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / ROOT.name
BOOT = "eb8bb39d-7518-4ae3-a871-854cc7ca33f2"
WARM = OUT / "night-cold-eb8bb39d/run.json"
assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == BOOT
started = time.monotonic()
while True:
    assert not (ROOT / "STOP_REQUESTED").exists(), "Experiments stopped"
    try:
        warm = json.loads(WARM.read_text())
        assert warm["boot_id"] == BOOT
        if "returncode" in warm:
            assert warm["returncode"] == 0 and "error" not in warm, "Cold load failed"
            break
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    assert time.monotonic() - started < 3700, "Cold load timeout"
    time.sleep(5)

sequence = [("model-night-a1-cpu", "cpu"), ("model-night-b1-common", "common"),
            ("model-night-c1-experts10", "experts10"), ("model-night-c2-experts10", "experts10"),
            ("model-night-b2-common", "common"), ("model-night-a2-cpu", "cpu")]
results = []
for label, mode in sequence:
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == BOOT
    assert not (ROOT / "STOP_REQUESTED").exists(), "Experiments stopped"
    folder = OUT / label
    if folder.exists():
        record = json.loads((folder / "run.json").read_text())
        assert record.get("returncode") == 0 and "error" not in record, "Inspect interrupted case before restarting"
    else:
        print("Starting", label, flush=True)
        with (OUT / (label + "-console.log")).open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "model_case.py"), "--label", label, "--mode", mode,
                            "--prompt", "64", "--tokens", "64", "--repetitions", "2", "--quiet"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    benchmarks = json.loads((folder / "stdout.txt").read_text())
    results.append({"label": label, "mode": mode, "benchmarks": benchmarks})
    (OUT / "night-model-sequence-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Completed", label, [(b["n_prompt"], b["n_gen"], b["avg_ts"]) for b in benchmarks], flush=True)
    time.sleep(15)
print("Quiet model comparison complete", flush=True)

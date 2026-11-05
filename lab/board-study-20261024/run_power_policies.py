#!/usr/bin/env python3
"""Compare four reversible CPU power policies, restoring the original settings."""
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from hardware_run import ROOT, LAB, OUT, run, sample, fix_owner


def settings():
    paths = list(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))
    paths += list(Path("/sys/devices/system/cpu").glob("cpu*/power/energy_perf_bias"))
    return {str(path): path.read_text().strip() for path in paths}


def apply(governor, epb):
    for path in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"):
        path.write_text(governor)
    for path in Path("/sys/devices/system/cpu").glob("cpu*/power/energy_perf_bias"):
        path.write_text(str(epb))
    current = settings()
    assert all(value == (governor if key.endswith("scaling_governor") else str(epb)) for key, value in current.items())


def main():
    memory = json.loads((OUT / "memory-96g/run.json").read_text())
    assert memory["returncode"] == 0 and "error" not in memory and "ECC" not in memory["sel_after"]
    original = settings()
    assert len(original) == 48
    (OUT / "power-policy-original.json").write_text(json.dumps(original, indent=2) + "\n")
    variants = {"A": ("schedutil", 7), "B": ("performance", 7), "C": ("performance", 0), "D": ("schedutil", 0)}
    assert all(value == ("schedutil" if key.endswith("scaling_governor") else "7") for key, value in original.items())
    command = [sys.executable, str(LAB / "v100-prep-20261024/launch.py"), "--mode", "cpu", "--bench"]
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for i, label in enumerate("ABCDDCBA"):
                governor, epb = variants[label]
                apply(governor, epb)
                print("policy", label, governor, epb, flush=True)
                idle = [sample()]
                for _ in range(5):
                    time.sleep(1)
                    idle.append(sample())
                report = run(f"policy-{i}-{label}", command, timeout=600)
                (OUT / f"policy-{i}-{label}" / "policy.json").write_text(json.dumps({"label": label,
                    "governor": governor, "epb": epb, "idle_samples": idle}, indent=2) + "\n")
        finally:
            for name, value in original.items():
                Path(name).write_text(value)
            restored = settings()
            (OUT / "power-policy-restored.json").write_text(json.dumps(restored, indent=2) + "\n")
            assert restored == original


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        main()
    finally:
        fix_owner()

#!/usr/bin/env python3
import fcntl
from pathlib import Path
import sys
from hardware_run import ROOT, LAB, OUT, ENV, run, fix_owner


def main():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    assert int(memory["MemAvailable"].split()[0]) >= 112 * 1024 * 1024
    prefix = [sys.executable, str(LAB / "without_thp.py"), "numactl", "--physcpubind=0-11", "--interleave=0,1",
              str(ROOT / "diagnostics/usr/bin/stressapptest")]
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for label, mib, seconds in (("memory-smoke", 4096, 20), ("memory-96g", 98304, 300)):
            command = prefix + ["-M", str(mib), "-s", str(seconds), "-m", "12", "-i", "2", "-W",
                                "--stop_on_errors", "--no_affinity", "--printsec", "30"]
            report = run(label, command, timeout=seconds + 240, keep_root=True)
            text = (OUT / label / "stdout.txt").read_text()
            assert "Status: PASS" in text


if __name__ == "__main__":
    try:
        main()
    finally:
        fix_owner()

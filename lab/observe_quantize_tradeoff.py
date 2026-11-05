#!/usr/bin/env python3
"""Attach counters only to the steady large-matrix loop, after data preparation."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from glm_q8_experiment import environment
from observe_glm_counters_light import snapshot
from run_quantize_tradeoff import LAB, OUT, BIN


def capture(kind):
    target = OUT / ("counters-" + kind)
    target.mkdir(exist_ok=False)
    command = ["numactl", "--physcpubind=0-23", "--interleave=0,1",
               str(BIN / "quantize-matvec"), "4096", "65536", "24", "20", "1", kind]
    report = {"command": command, "counter_seconds": 15, "telemetry": [], "counters": []}
    child, processes, logs, observer = None, [], [], None

    def collect(pid):
        started = time.monotonic()
        try:
            for second in range(16):
                time.sleep(max(0, started + second - time.monotonic()))
                report["telemetry"].append(snapshot(pid))
            report["child_alive_at_window_end"] = child.poll() is None
        except Exception as error:
            report["observer_error"] = repr(error)

    try:
        with (target / "stdout.txt").open("w") as stdout, (target / "stderr.txt").open("w") as stderr:
            child = subprocess.Popen(command, env=environment(), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=stderr, text=True, start_new_session=True)
            for line in child.stdout:
                stdout.write(line); stdout.flush()
                row = json.loads(line)
                if not row.get("ready"):
                    continue
                assert not processes
                report["pid"] = child.pid
                events = ["task-clock", "cycles:u", "instructions:u", "context-switches",
                          "cpu-migrations", "minor-faults", "major-faults", "duration_time",
                          "cycle_activity.stalls_total:u", "cycle_activity.stalls_l2_pending:u"]
                specs = [
                    ("core", ["-p", str(child.pid), "-e", ",".join(events)]),
                    ("imc", ["-a", "-A", "-I", "1000", "-e",
                             "uncore_imc_*/event=0x04,umask=0x03/,uncore_imc_*/event=0x04,umask=0x0c/"]),
                ]
                for name, options in specs:
                    args = ["perf", "stat", "-x", ",", "--no-big-num", "-o", str(target / (name + ".csv")),
                            *options, "--", "sleep", "15"]
                    log = (target / (name + ".stderr.txt")).open("w"); logs.append(log)
                    processes.append(subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True))
                    report["counters"].append({"name": name, "command": args})
                observer = threading.Thread(target=collect, args=(child.pid,)); observer.start()
            report["exit_code"] = child.wait(timeout=5)
            for proc, entry in zip(processes, report["counters"]):
                entry["exit_code"] = proc.wait(timeout=5)
            if observer:
                observer.join(timeout=5)
            assert report["exit_code"] == 0 and len(processes) == 2
            assert len(report["telemetry"]) == 16 and report["child_alive_at_window_end"]
            assert all(p.returncode == 0 for p in processes)
            report["valid"] = True
    finally:
        for process in processes + ([child] if child else []):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        for log in logs: log.close()
        (target / "observations.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"done": str(target), "valid": report["valid"]}), flush=True)


def main():
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert json.loads((OUT / "complete.json").read_text())["complete"]
        for kind in ("q8_0", "q6_K", "q4_K", "q3_K"):
            print("Capturing " + kind, flush=True)
            capture(kind)


if __name__ == "__main__":
    main()

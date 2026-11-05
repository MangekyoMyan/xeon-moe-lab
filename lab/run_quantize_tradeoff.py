#!/usr/bin/env python3
"""Serial CPU-only quantization kernel comparisons with identical dimensions."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from bench_deepseek_light import snapshot
from glm_q8_experiment import environment, temperatures

LAB = Path(__file__).resolve().parent
OUT = LAB / "results/quantize-tradeoff-20261021"
BIN = LAB / "quantize-tradeoff-20261021"
TYPES = ["q8_0", "q6_K", "q5_K", "q4_K", "q3_K", "q2_K", "q4_0", "iq4_xs"]


def run(label, command, timeout=1200):
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    report = {"label": label, "command": command, "started_unix": time.time(),
              "observations": [], "environment": environment(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    # Retain only benchmark controls, not unrelated inherited environment values.
    report["environment"] = {k: v for k, v in report["environment"].items()
                             if k.startswith(("OMP_", "GGML_", "OPENBLAS"))}
    start = time.monotonic()
    child = None
    try:
        with (folder / "stdout.txt").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
            child = subprocess.Popen(command, env=environment(), stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            report["pid"] = child.pid
            progress = 0
            while child.poll() is None:
                elapsed = time.monotonic() - start
                row = snapshot(child.pid)
                if row:
                    row["elapsed_s"] = elapsed
                    row["cpu_temperatures_C"] = temperatures()
                    report["observations"].append(row)
                    if row["cpu_temperatures_C"] and max(row["cpu_temperatures_C"].values()) >= 85:
                        raise RuntimeError("CPU temperature cutoff")
                if elapsed > timeout:
                    raise RuntimeError("Measurement timeout")
                if elapsed >= progress:
                    print(json.dumps({"running": label, "seconds": round(elapsed)}), flush=True)
                    (folder / "run.json").write_text(json.dumps(report, indent=2) + "\n")
                    progress = elapsed + 30
                time.sleep(1)
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            report["exit_code"] = child.wait()
        report["elapsed_s"] = time.monotonic() - start
        (folder / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["exit_code"] == 0, (folder / "stderr.txt").read_text()[-2000:]
    print(json.dumps({"done": label, "elapsed_s": report["elapsed_s"]}), flush=True)
    return report


def main():
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        one_cpu = ["numactl", "--physcpubind=0", "--membind=0"]
        dual_cpu = ["numactl", "--physcpubind=0-23", "--interleave=0,1"]
        run("matvec-validation", dual_cpu + [str(BIN / "quantize-matvec"), "4096", "24", "24", "0.03", "1"])
        for variant in ("original", "fixed"):
            run(f"stock-q8-{variant}", one_cpu + [str(BIN / f"test-quantize-perf-{variant}"),
                "--type", "q8_0", "--op", "vec_dot_q", "--size", "65536", "-i", "500"])
        for repeat in range(2):
            for regime, size, count in (("cache", "4096", "10000"), ("stream", "33554432", "20")):
                args = [arg for kind in TYPES for arg in ("--type", kind)]
                run(f"quant-perf-{regime}-{repeat}", one_cpu + [str(BIN / "test-quantize-perf-fixed"),
                    *args, "--op", "vec_dot_q", "--op", "dequantize_row_q",
                    "--size", size, "-i", count])
        for threads in (1, 12, 24):
            for regime, rows in (("cache", 24), ("stream", 65536)):
                run(f"matvec-k4096-{regime}-t{threads}", dual_cpu + [str(BIN / "quantize-matvec"),
                    "4096", str(rows), str(threads), "0.25", "6"])
        run("matvec-k1408-stream-t24", dual_cpu + [str(BIN / "quantize-matvec"),
            "1408", "196608", "24", "0.25", "6"])
        (OUT / "complete.json").write_text(json.dumps({"complete": True}) + "\n")


if __name__ == "__main__":
    main()

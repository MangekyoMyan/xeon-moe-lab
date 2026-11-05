#!/usr/bin/env python3
"""Serial GLM Q8 experiments with bounded runtime and light host telemetry."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from bench_deepseek_light import snapshot

LAB = Path(__file__).resolve().parent
OUT = LAB / "results/glm-air-q8-20261020"
MODEL = LAB / "models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf"
BUILDS = {"A": LAB / "build-avx-deep", "B": LAB / "build-avx-prefetch"}


def temperatures():
    values = {}
    for folder in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if folder.joinpath("name").read_text().strip() != "coretemp":
                continue
            for path in folder.glob("temp*_input"):
                values[str(path)] = int(path.read_text()) / 1000
        except OSError:
            continue
    return values


def environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GGML_", "LLAMA_ARG_", "OMP_", "GOMP_"))
           and k != "LD_LIBRARY_PATH"}
    env.update(LC_ALL="C", OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24",
               OMP_MAX_ACTIVE_LEVELS="1", OPENBLAS_NUM_THREADS="1",
               GGML_CPU_NUMA_FULL="0", GGML_CPU_NUMA_REPLICA="0")
    return env


def run(label, build="B", threads="12", tokens=128, prompt=0, repetitions=3,
        depth=0, response=None, timeout=2400, load="mmap", batch=128,
        extra=None, extra_env=None):
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=False)
    binary = BUILDS.get(build, Path(build)) / "bin" / (
        "llama-completion" if response is not None else "llama-bench")
    env = environment()
    env.update(extra_env or {})
    command = ["numactl", "--physcpubind=0-23", "--interleave=0,1",
               str(binary), "-m", str(MODEL), "-t", str(threads),
               "-b", str(batch), "-ub", str(batch), "-ngl", "0",
               "-lm", load, "--cpu-mask", "0xffffff", "--cpu-strict", "1",
               "--offline"]
    if response is None:
        command += ["-p", str(prompt), "-n", str(tokens), "-d", str(depth),
                    "-r", str(repetitions), "-o", "json", "--progress",
                    "-dev", "none", "-nkvo", "1", "-nopo", "1", "--repack", "0",
                    "-lzm", "off"]
    else:
        command += ["-tb", "24", "-c", "4096", "-n", str(tokens),
                    "--cpu-mask-batch", "0xffffff", "--cpu-strict-batch", "1",
                    "--no-repack", "--jinja", "--reasoning", "off",
                    "--temp", "0", "--seed", "42", "--no-display-prompt",
                    "--single-turn", "--simple-io", "--color", "off", "--perf", "-p", response]
    command += extra or []
    artifacts = [binary.resolve()] + [p for p in binary.parent.glob("*.so*") if not p.is_symlink()]
    report = {
        "label": label, "command": command, "build": str(binary.parent.parent),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "started_unix": time.time(), "environment": {k: v for k, v in env.items()
            if k.startswith(("GGML_", "OMP_", "GOMP_", "OPENBLAS"))},
        "artifacts_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        "observations": [], "phases": [],
        "caveat": "Synthetic random-token benchmark unless response is set. mmap NUMA policy does not relocate previously cached file pages.",
    }
    start = time.monotonic()
    offset = 0
    progress = 0
    child = None

    def save():
        (folder / "run.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(json.dumps({"start": label, "command": command}), flush=True)
    with (folder / "stdout.txt").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                                     stderr=stderr, env=env, start_new_session=True)
            report["pid"] = child.pid
            while child.poll() is None:
                elapsed = time.monotonic() - start
                row = snapshot(child.pid)
                if row:
                    row["elapsed_seconds"] = elapsed
                    report["observations"].append(row)
                with (folder / "stderr.txt").open() as log:
                    log.seek(offset)
                    for line in log:
                        if "llama-bench: benchmark" in line or "error" in line.lower():
                            phase = {"elapsed_seconds": elapsed, "unix_time": time.time(),
                                     "message": line.strip(), "snapshot": row}
                            report["phases"].append(phase)
                            print(json.dumps({"phase": phase}), flush=True)
                    offset = log.tell()
                if elapsed >= progress:
                    temps = temperatures()
                    if row:
                        row["cpu_temperatures_c"] = temps
                    print(json.dumps({"label": label, "elapsed": round(elapsed),
                                      "sample": row}), flush=True)
                    save()
                    progress = elapsed + 30
                    if temps and max(temps.values()) >= 85:
                        raise RuntimeError("CPU temperature reached experiment cutoff 85 C")
                if row and row["available_kib"] < 1536 * 1024:
                    raise RuntimeError("Available RAM fell below experiment floor 1.5 GiB")
                if elapsed > timeout:
                    raise RuntimeError("Experiment timeout")
                time.sleep(1)
        except BaseException as exc:
            report["error"] = str(exc)
            raise
        finally:
            if child is not None:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                report["exit_code"] = child.wait()
            report["elapsed_seconds"] = time.monotonic() - start
            save()
    if report["exit_code"] != 0:
        raise RuntimeError((folder / "stderr.txt").read_text()[-4000:])
    if response is None:
        report["benchmarks"] = json.loads((folder / "stdout.txt").read_text())
        print(json.dumps({"done": label, "results": [
            {k: b[k] for k in ("n_threads", "n_prompt", "n_gen", "avg_ts", "stddev_ts")}
            for b in report["benchmarks"]]}), flush=True)
    else:
        print(json.dumps({"done": label, "response": (folder / "stdout.txt").read_text(),
                          "timing": (folder / "stderr.txt").read_text()[-1800:]}), flush=True)
    save()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--build", default="B")
    ap.add_argument("--threads", default="12")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--response")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--load", default="mmap")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(**vars(args))


if __name__ == "__main__":
    main()

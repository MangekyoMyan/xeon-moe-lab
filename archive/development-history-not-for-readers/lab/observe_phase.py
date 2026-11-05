#!/usr/bin/env python3
"""Capture an approximate sample inside one timed llama-bench loop, CPU only.

This is a diagnostic run, not an A/B throughput benchmark.  The stderr marker
is received through a pipe; it is not an acknowledged phase boundary.  A sample
is accepted only after the benchmark completes and its own timed duration
exceeds the observed window end by at least two seconds.  This conservative
check cannot prove exact synchronization or isolate system-wide IMC traffic.

No model download, sudo, sysctl change, cache eviction, or GPU use is performed.
Use the original Xeon server and its CPU-only build.  perf.data is a private
diagnostic artifact; share the symbol summary if the full capture is unwanted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time


ERROR_LINE = re.compile(r"\b(error|failed|fatal|assertion)\b|terminate called|segmentation fault", re.I)
CORE_EVENTS = "task-clock,cycles:u,instructions:u,context-switches,cpu-migrations,minor-faults,major-faults"


def id_set(value):
    result = set()
    for piece in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", piece):
            raise ValueError("CPU/node lists must contain integers and ranges, for example 0-23 or 0,1")
        bounds = [int(v) for v in piece.split("-")]
        first, last = bounds[0], bounds[-1]
        if not 0 <= first <= last <= 65535:
            raise ValueError("Invalid CPU/node range")
        result.update(range(first, last + 1))
    if not result:
        raise ValueError("An empty CPU/node list is not allowed")
    return result


def snapshot(pid):
    data = {"monotonic": time.monotonic()}
    for name in ("io", "status"):
        try:
            rows = Path(f"/proc/{pid}/{name}").read_text().splitlines()
            keep = {"read_bytes", "write_bytes", "VmRSS", "VmHWM", "RssAnon", "RssFile", "Cpus_allowed_list"}
            data[name] = {k: v.strip() for row in rows if ":" in row for k, v in [row.split(":", 1)] if k in keep}
        except OSError:
            data[name] = None
    return data


def stop_group(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)
    except ProcessLookupError:
        proc.wait(timeout=3)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path, help="Existing first GGUF shard")
    ap.add_argument("--phase", required=True, choices=("prefill", "decode"))
    ap.add_argument("--mode", choices=("record", "stat", "wait", "memory", "futex"), default="record")
    ap.add_argument("--threads", type=int, help="Default: prefill 24, decode 12")
    ap.add_argument("--tokens", type=int, help="Default: prefill 512, decode 128; must outlast sample window")
    ap.add_argument("--depth", type=int, default=0, help="Untimed prefix; default 0")
    ap.add_argument("--cpus", required=True, help="Explicit target CPU list; Xeon example: 0-23")
    ap.add_argument("--nodes", required=True, help="Explicit interleave nodes; Xeon example: 0,1")
    ap.add_argument("--seconds", type=float, default=10.0, help="Sample window, 0 < seconds <= 60")
    ap.add_argument("--timeout", type=float, default=1200.0, help="Total benchmark timeout, including load and warmup")
    ap.add_argument("--output", required=True, type=Path, help="New directory; existing directories are refused")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only; no executable/model/topology checks")
    a = ap.parse_args()
    a.threads = a.threads if a.threads is not None else (24 if a.phase == "prefill" else 12)
    a.tokens = a.tokens if a.tokens is not None else (512 if a.phase == "prefill" else 128)
    try:
        cpus, nodes = id_set(a.cpus), id_set(a.nodes)
    except ValueError as exc:
        ap.error(str(exc))
    if not 1 <= a.threads <= len(cpus) or not 1 <= a.tokens or not 0 <= a.depth:
        ap.error("Require 1 <= threads <= selected CPUs, tokens > 0, depth >= 0")
    if max(cpus) >= 512:
        ap.error("This helper supports CPU mask bits 0 through 511")
    if not 0 < a.seconds <= 60 or not a.seconds + 2 < a.timeout <= 7200:
        ap.error("Require 0 < seconds <= 60 and seconds + 2 < timeout <= 7200")
    a.bench, a.model, a.output = (p.expanduser().resolve() for p in (a.bench, a.model, a.output))
    return a, cpus, nodes


def main():
    a, cpus, nodes = parse_args()
    mask = hex(sum(1 << cpu for cpu in cpus))
    command = ["numactl", f"--physcpubind={a.cpus}", f"--interleave={a.nodes}", str(a.bench),
               "-m", str(a.model), "-p", str(a.tokens if a.phase == "prefill" else 0),
               "-n", str(a.tokens if a.phase == "decode" else 0), "-d", str(a.depth),
               "-t", str(a.threads), "-b", "512", "-ub", "512", "--poll", "50",
               "-ngl", "0", "-dev", "none", "-nkvo", "1", "-nopo", "1",
               "-lm", "mmap", "--cpu-mask", mask, "--cpu-strict", "1",
               "-r", "1", "-o", "json", "--offline", "--progress"]

    def perf_command(pid):
        if a.mode == "wait":
            return ["sleep", str(a.seconds)]
        if a.mode == "futex":
            return ["perf", "record", "-e", "syscalls:sys_enter_futex", "-e", "syscalls:sys_exit_futex", "-o", str(a.output / "profile.data"), "-p", str(pid), "--", "sleep", str(a.seconds)]
        if a.mode == "memory":
            return ["perf", "mem", "record", "-U", "-t", "load", "--ldlat", "30", "-c", "1000", "-o", str(a.output / "profile.data"), "--pid", str(pid), "--", "sleep", str(a.seconds)]
        if a.mode == "record":
            prefix = ["perf", "record", "-e", "cycles:u", "-F", "99", "-g", "-o", str(a.output / "profile.data")]
        else:
            prefix = ["perf", "stat", "-x", ",", "--no-big-num", "-e", CORE_EVENTS,
                      "-o", str(a.output / "profile.csv")]
        return prefix + ["-p", str(pid), "--", "sleep", str(a.seconds)]

    report = {"status": "invalid", "valid_sample": False,
              "method": "Approximate timed-loop window; marker receipt has unbounded scheduling latency, not an exact phase boundary.",
              "throughput_use": "Diagnostic timing includes profiling overhead; exclude from A/B performance results.",
              "callchains": "Default frame-pointer callchains may be incomplete; inspect flat symbols." if a.mode == "record" else "No callchains captured in this mode.",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "command": command, "perf_command": perf_command("BENCH_PID"), "phases": []}
    if a.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    a.output.mkdir(parents=True, exist_ok=False)
    bench = profiler = None
    handles = []
    sel = selectors.DefaultSelector()
    started = time.monotonic()
    marker_at = profile_started = profile_finished = None
    target = f"llama-bench: benchmark 1/1: {'prompt' if a.phase == 'prefill' else 'generation'} run 1/1"
    buffered = b""
    eof = False
    env = dict(os.environ, LC_ALL="C")
    report["environment"] = {k: v for k, v in env.items() if k in (
        "LC_ALL", "LD_LIBRARY_PATH", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_NUM_THREADS",
        "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "GGML_CPU_TILED_MM", "GGML_CPU_TILED_MM_FORCE")}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for executable in ("numactl", "perf", "sleep", "true"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"Required executable unavailable: {executable}; no installation or privilege changes were attempted")
        if not a.bench.is_file() or not os.access(a.bench, os.X_OK) or not a.model.is_file():
            raise RuntimeError("The CPU-only benchmark executable and existing first model shard must be available")
        if not cpus <= os.sched_getaffinity(0):
            raise RuntimeError("Requested CPUs are outside this process's allowed CPU set; verify the target host topology")
        online_nodes = Path("/sys/devices/system/node/online")
        if online_nodes.exists() and not nodes <= id_set(online_nodes.read_text().strip()):
            raise RuntimeError("Requested NUMA nodes are not online")
        report["bench_sha256"] = hashlib.sha256(a.bench.read_bytes()).hexdigest()
        report["model_shard_bytes"] = a.model.stat().st_size
        try:
            report["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            report["boot_id"] = None
        preflight = ["perf", "stat", "-e", "cycles:u", "-o", str(a.output / "perf-preflight.log"), "--", "true"]
        check = subprocess.run(preflight, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=15)
        report["perf_preflight"] = {"command": preflight, "exit_code": check.returncode,
                                   "stderr": check.stderr.decode(errors="replace")}
        if check.returncode:
            raise RuntimeError("perf preflight failed; inspect perf-preflight.log for permissions/event support. No sudo or sysctl changes were attempted")

        out = (a.output / "bench.json").open("wb")
        log = (a.output / "bench.stderr.log").open("wb")
        handles.extend([out, log])
        bench = subprocess.Popen(command, stdout=out, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                 env=env, start_new_session=True)
        report["pid"] = bench.pid
        sel.register(bench.stderr, selectors.EVENT_READ)
        print(f"Waiting for the timed {a.phase} loop; output: {a.output}", flush=True)

        while not (bench.poll() is not None and eof):
            if time.monotonic() - started > a.timeout:
                raise RuntimeError("Total timeout, including load/warmup/depth; sample invalid")
            for key, _ in sel.select(timeout=0.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    eof = True
                    if buffered:
                        chunk, buffered = b"\n", buffered
                    else:
                        continue
                else:
                    log.write(chunk)
                    log.flush()
                buffered += chunk
                while b"\n" in buffered:
                    raw_line, buffered = buffered.split(b"\n", 1)
                    line = raw_line.decode(errors="replace").strip()
                    now = time.monotonic()
                    if ERROR_LINE.search(line):
                        raise RuntimeError(f"Benchmark error: {line[:500]}")
                    if line.startswith("llama-bench: benchmark"):
                        report["phases"].append({"elapsed": now - started, "message": line})
                        if marker_at is not None:
                            raise RuntimeError("Unexpected additional phase after the selected timed loop; sample invalid")
                    if line == target:
                        marker_at = now
                        report["window_start_snapshot"] = snapshot(bench.pid)
                        perfout = (a.output / "perf.stdout.log").open("wb")
                        perferr = (a.output / "perf.stderr.log").open("wb")
                        handles.extend([perfout, perferr])
                        pcommand = perf_command(bench.pid)
                        profile_started = time.monotonic()
                        profiler = subprocess.Popen(pcommand, stdout=perfout, stderr=perferr,
                                                    stdin=subprocess.DEVNULL, env=env, start_new_session=True)
                        report["perf_command"] = pcommand
                        report["marker_elapsed"] = marker_at - started
                        report["attach_start_offset"] = profile_started - marker_at
                        print(f"Capturing a {a.seconds:g}-second {a.mode} window; warmup/depth markers did not trigger capture", flush=True)

            if profiler is not None and profile_finished is None:
                code = profiler.poll()
                if code is not None:
                    profile_finished = time.monotonic()
                    report["perf_exit_code"] = code
                    report["window_end_offset"] = profile_finished - marker_at
                    report["window_end_snapshot"] = snapshot(bench.pid)
                    if code:
                        raise RuntimeError("perf attach/capture failed; inspect perf.stderr.log and profile.csv if present")
                    if profile_finished - profile_started + 0.2 < a.seconds:
                        raise RuntimeError("perf ended before the requested window elapsed; sample invalid")
                elif time.monotonic() - profile_started > a.seconds + 5:
                    raise RuntimeError("perf exceeded the bounded capture window; sample invalid")
            if bench.poll() is not None and profile_finished is None:
                raise RuntimeError("Benchmark ended before the profiling window completed, or before the target marker")

        report["bench_exit_code"] = bench.wait()
        out.flush()
        if report["bench_exit_code"] != 0:
            raise RuntimeError("Benchmark exited unsuccessfully; sample invalid")
        if marker_at is None or profile_finished is None:
            raise RuntimeError("Expected timed-loop marker or completed profiling window was not observed")
        profile_path = a.output / ("profile.data" if a.mode in ("record", "memory", "futex") else "profile.csv")
        if a.mode != "wait" and (not profile_path.is_file() or profile_path.stat().st_size == 0):
            raise RuntimeError("perf did not create a nonempty profile")
        if a.mode == "stat" and re.search(r"<not (?:counted|supported)>", profile_path.read_text()):
            raise RuntimeError("One or more requested counters were not counted/supported; sample invalid")
        rows = json.loads((a.output / "bench.json").read_text())
        if len(rows) != 1 or len(rows[0].get("samples_ns", [])) != 1:
            raise RuntimeError("Expected exactly one benchmark row and one timed repetition")
        row = rows[0]
        expected = {"n_prompt": a.tokens if a.phase == "prefill" else 0,
                    "n_gen": a.tokens if a.phase == "decode" else 0, "n_depth": a.depth, "n_threads": a.threads}
        if any(row.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Benchmark result does not match the requested workload")
        report["bench_timed_seconds"] = row["samples_ns"][0] / 1e9
        report["nominal_margin_seconds"] = report["bench_timed_seconds"] - report["window_end_offset"]
        if report["nominal_margin_seconds"] < 2:
            raise RuntimeError("Window end plus the two-second margin is outside the timed loop; increase tokens or shorten the window")
        report.update(status="valid_approximate_sample", valid_sample=True)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        report["reason"] = str(exc) or type(exc).__name__
    finally:
        stop_group(profiler)
        stop_group(bench)
        if bench is not None:
            report["bench_exit_code"] = bench.returncode
            if bench.stderr is not None:
                bench.stderr.close()
        if profiler is not None:
            report["perf_exit_code"] = profiler.returncode
        sel.close()
        for handle in handles:
            handle.close()
        report["wall_seconds"] = time.monotonic() - started
        (a.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "reason": report.get("reason"),
                      "report": str(a.output / "capture.json")}), flush=True)
    return 0 if report["valid_sample"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Observe the existing fine-interleave 26B experiment without engine changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parent
OUT = LAB / "results/bottleneck-melody-fine-20261020"
BENCH = LAB / "review-full-replica-20261020/build/bin/llama-bench"
MODEL = LAB / "models/mradermacher--Melody1437-26B-A4B-i1-GGUF/Melody1437-26B-A4B.i1-Q6_K.gguf"


def snapshot():
    files = [
        LAB / "run_melody.sh",
        LAB / "run_deepseek_flash.sh",
        LAB / "build-avx-prefetch/bin/libggml-cpu.so",
        LAB / "review-full-replica-20261020/build/bin/libggml-cpu.so",
        BENCH,
        LAB / "observe_phase.py",
        LAB / "observe_full_replica.py",
        LAB / "observe_melody_limits.py",
        Path(__file__).resolve(),
    ]
    settings = [
        "/proc/sys/kernel/random/boot_id",
        "/proc/sys/kernel/numa_balancing",
        "/proc/sys/kernel/sched_schedstats",
        "/sys/kernel/mm/transparent_hugepage/enabled",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
    ]
    return {
        "unix_time": time.time(),
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "settings": {p: Path(p).read_text().strip() for p in settings},
    }


def main():
    assert "THP_enabled:\t0" in Path("/proc/self/status").read_text()
    OUT.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # Make the inference configuration explicit and reproducible.
    for k in list(env):
        if k.startswith(("OMP_", "GOMP_", "GGML_CPU_")) or k == "LD_LIBRARY_PATH":
            env.pop(k)
    env.update(
        OMP_DYNAMIC="FALSE",
        OMP_THREAD_LIMIT="24",
        OMP_MAX_ACTIVE_LEVELS="1",
        OPENBLAS_NUM_THREADS="1",
        GGML_CPU_NUMA_REPLICA="0",
        GGML_CPU_NUMA_REPLICA_TRACE="0",
        GGML_CPU_NUMA_FULL="2",
        LC_ALL="C",
    )
    report = {
        "complete": False,
        "before": snapshot(),
        "environment": {k: v for k, v in env.items()
                        if k.startswith(("OMP_", "GOMP_", "GGML_CPU_", "OPENBLAS_"))},
        "process_thp_disabled": True,
        "purpose": "Diagnostic profiling, not a new throughput A/B or engine optimization.",
        "commands": [],
    }
    common = ["--bench", str(BENCH), "--model", str(MODEL),
              "--threads", "24", "--tokens", "512", "--depth", "0"]
    commands = [
        [sys.executable, str(LAB / "observe_melody_limits.py"),
         "decode", "stalls", *common, "--window", "18",
         "--output", str(OUT / "stalls")],
    ]
    for mode in ("record", "memory"):
        commands.append(
            [sys.executable, str(LAB / "observe_phase.py"),
             *common, "--phase", "decode", "--mode", mode, "--cpus", "0-23",
             "--nodes", "0,1", "--seconds", "18", "--output", str(OUT / mode)]
        )
    try:
        for i, cmd in enumerate(commands):
            print(f"Diagnostic {i + 1}/{len(commands)}: {cmd[-1]}", flush=True)
            report["commands"].append({"argv": cmd, "started": time.time()})
            result = subprocess.run(cmd, env=env, check=False)
            report["commands"][-1].update(exit_code=result.returncode, finished=time.time())
            if result.returncode:
                raise RuntimeError(f"Observation {i + 1} failed")
        report["complete"] = True
    finally:
        report["after"] = snapshot()
        report["normal_artifacts_and_global_settings_unchanged"] = (
            report["before"]["sha256"] == report["after"]["sha256"]
            and report["before"]["settings"] == report["after"]["settings"]
        )
        (OUT / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

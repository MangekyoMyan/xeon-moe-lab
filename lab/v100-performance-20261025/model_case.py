#!/usr/bin/env python3
"""Run one controlled GLM placement case using the same CUDA-enabled binary."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--mode", choices=["cpu", "common", "experts10"], required=True)
ap.add_argument("--prompt", default="128")
ap.add_argument("--tokens", default="128")
ap.add_argument("--depth", default="0")
ap.add_argument("--repetitions", default="2")
ap.add_argument("--threads", default="24")
ap.add_argument("--poll", type=int, default=50)
ap.add_argument("--op-offload", choices=["on", "off"], default="on")
ap.add_argument("--graphs", choices=["on", "off"], default="on")
ap.add_argument("--timeout", default="1200")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--response")
args = ap.parse_args()
assert 0 <= args.poll <= 100
binary = LAB / "v100-prep-20261024/build-cuda/bin" / ("llama-completion" if args.response else "llama-bench")
model = LAB / "models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf"
command = [sys.executable, str(LAB / "without_thp.py"), "numactl", "--physcpubind=0-23", "--interleave=0,1",
           str(binary), "-m", str(model), "-t", args.threads, "-b", "128", "-ub", "128",
           "--cpu-mask", "0xffffff", "--cpu-strict", "1", "--poll", str(args.poll), "--offline"]
if args.response:
    command += ["--load-mode", "mmap", "--lazy-mode", "off", "--no-repack", "-c", "2048",
                "-tb", args.threads, "--cpu-mask-batch", "0xffffff", "--cpu-strict-batch", "1",
                "--fit", "off", "--jinja", "--reasoning", "off", "--temp", "0", "--seed", "42",
                "--single-turn", "--simple-io", "--color", "off", "--no-display-prompt", "--perf",
                "-n", args.tokens, "-p", args.response]
    if args.mode == "cpu":
        command += ["-ngl", "0", "--device", "none", "--no-kv-offload", "--no-op-offload"]
    else:
        command += ["-ngl", "all", "--n-cpu-moe", "46" if args.mode == "common" else "36", "-fa", "auto"]
        if args.op_offload == "off":
            command += ["--no-op-offload"]
else:
    command += ["-lm", "mmap", "-lzm", "off", "--repack", "0", "-p", args.prompt, "-n", args.tokens,
                "-d", args.depth, "-r", args.repetitions, "--progress", "-o", "json"]
    if args.mode == "cpu":
        command += ["-ngl", "0", "-dev", "none", "-nkvo", "1", "-nopo", "1"]
    else:
        command += ["-ngl", "99", "-ncmoe", "46" if args.mode == "common" else "36", "-fa", "auto"]
        if args.op_offload == "off":
            command += ["-nopo", "1"]
watch = [sys.executable, str(ROOT / "watch.py"), "--label", args.label, "--timeout", args.timeout]
if args.quiet:
    watch += ["--quiet"]
if args.graphs == "off":
    watch += ["--env", "GGML_CUDA_DISABLE_GRAPHS=1"]
os.execv(sys.executable, [*watch, "--", *command])

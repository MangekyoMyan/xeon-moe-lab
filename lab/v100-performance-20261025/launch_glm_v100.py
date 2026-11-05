#!/usr/bin/env python3
"""Launch the validated, low-noise GLM CPU/PG500 split."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
sys.path.insert(0, str(LAB / "v100-prep-20261024"))
from probe import inspect
from launch import require_policy, select_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--context", type=int, default=4096)
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--threads", type=int, default=12)
parser.add_argument("--threads-batch", type=int)
args, extra = parser.parse_known_args()
if extra[:1] == ["--"]:
    extra = extra[1:]
batch_threads = args.threads_batch or args.threads
if args.context <= 0 or args.max_tokens <= 0 or not (1 <= args.threads <= 24 and 1 <= batch_threads <= 24):
    parser.error("Invalid context, token limit, or thread count")
report = inspect()
gpu_id = "GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3"
gpu = next((d for d in report.get("nvidia_smi", []) if d["uuid"] == gpu_id), None)
if gpu is None or float(gpu["temperature"].split()[0]) >= 55:
    parser.error("Expected PG500 is unavailable or already too warm")
def normalized_bus(value):
    domain, bus, function = value.lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{function}"
matched = next((d for d in report.get("cuda", {}).get("devices", [])
                if normalized_bus(d["pci_bus_id"]) == normalized_bus(gpu["bus_id"])), None)
if matched is None:
    parser.error("Expected PG500 is not available through CUDA")
device = select_gpu(report, matched["index"])
require_policy(device)
model = LAB / "models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf"
if not model.is_file():
    parser.error("The retained GLM model is missing")
env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_ARG_", "OMP_", "GOMP_", "CUDA_")) and k != "LD_LIBRARY_PATH"}
env.update(LC_ALL="C", OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24", OMP_MAX_ACTIVE_LEVELS="1",
           OPENBLAS_NUM_THREADS="1", GGML_CPU_NUMA_FULL="0", GGML_CPU_NUMA_REPLICA="0", CUDA_VISIBLE_DEVICES=gpu_id)
command = [sys.executable, str(LAB / "without_thp.py"), "numactl", "--physcpubind=0-23", "--interleave=0,1",
           str(LAB / "v100-prep-20261024/build-cuda/bin/llama-completion"), "-m", str(model),
           "-t", str(args.threads), "-tb", str(batch_threads), "--poll", "0", "--poll-batch", "0",
           "--cpu-mask", "0xffffff", "--cpu-strict", "1", "--cpu-mask-batch", "0xffffff", "--cpu-strict-batch", "1",
           "-c", str(args.context), "-b", "128", "-ub", "128", "-n", str(args.max_tokens),
           "--load-mode", "mmap", "--lazy-mode", "off", "--no-repack", "-ngl", "all", "--n-cpu-moe", "36",
           "--no-op-offload", "-fa", "auto", "--fit", "off", "--offline", "--jinja", "--reasoning", "off",
           "--temp", "0", "--seed", "42", "--single-turn", "--simple-io", "--color", "off", "--no-display-prompt", "--perf",
           *extra]
if args.dry_run:
    print(json.dumps({"command": command, "GPU": gpu["name"], "power_limit": gpu["power_limit"],
                      "ECC": gpu["ecc_current"], "scope": "Static placement: shared weights and 10 expert layers on GPU; remaining experts on CPU."}, indent=2))
else:
    os.execvpe(command[0], command, env)

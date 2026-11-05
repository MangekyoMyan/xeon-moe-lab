#!/usr/bin/env python3
"""Run one bounded NVFP4 conversation using the validated private loader."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
lab = root.parent
base = lab / "v100-prep-20261024"
ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["glm", "qwen"])
ap.add_argument("prompt")
ap.add_argument("--tokens", type=int, default=256)
args = ap.parse_args()
if not 1 <= args.tokens <= 512:
    ap.error("--tokens must be between 1 and 512")
if len(args.prompt.encode()) > 4096:
    ap.error("Keep this bounded smoke-test prompt under 4096 bytes")
manifest = json.loads((root / "loader-build.json").read_text())
for path, expected in manifest["sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Runtime or source changed: " + path
models = json.loads((root / "download-state.json").read_text())["models"]
model = models[0 if args.model == "qwen" else 1]
assert model["status"] == "verified"
assert Path(model["path"]).stat().st_size == model["bytes"]
label = "manual-" + args.model + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
command = ["env", "LD_LIBRARY_PATH=" + ":".join(map(str, [root / "runtime", base / "build-cuda/bin", base / "cuda-12.9.2/lib"])),
    sys.executable, str(lab / "without_thp.py"), "numactl", "--physcpubind=6-11,18-23", "--membind=1",
    str(base / "build-cuda/bin/llama-completion"), "-m", model["path"], "-t", "6", "-tb", "6",
    "-b", "128", "-ub", "128", "--offline", "-lm", "mmap", "--no-repack", "-c", "2048",
    "-n", str(args.tokens), "-ngl", "99", "-fa", "auto", "--jinja", "-cnv", "-st", "-rea", "off",
    "--temp", "0", "--seed", "42", "--simple-io", "--no-display-prompt", "-p", args.prompt]
guard = [sys.executable, str(root / "watch.py"), "--label", label, "--timeout", "600",
         "--env", "CUDA_VISIBLE_DEVICES=GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3", "--", *command]
rc = subprocess.run(([] if os.geteuid() == 0 else ["sudo", "-n"]) + guard).returncode
folder = lab / "results" / root.name / label
if (folder / "stdout.txt").exists():
    print((folder / "stdout.txt").read_text())
print("Logs:", folder)
sys.exit(rc)

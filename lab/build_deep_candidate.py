#!/usr/bin/env python3
"""Build a prepared isolated variant with flags matching the pool baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = args.directory.resolve()
source, build = root / "source", root / "build"
commands = [
    ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
     "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DGGML_OPENMP=ON", "-DGGML_BLAS=OFF",
     "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_OPENCL=OFF", "-DGGML_METAL=OFF",
     "-DGGML_SYCL=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_BUILD_SERVER=OFF",
     "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_BUILD_COMMON=ON", "-DLLAMA_BUILD_TOOLS=ON", "-DGGML_NATIVE=ON"],
    ["cmake", "--build", str(build), "--target", "llama-bench", "llama-completion",
     "test-backend-ops", "test-quantize-fns", "-j", "8"],
]
records = []
for name, command in zip(("configure", "build"), commands):
    start = time.monotonic()
    print(json.dumps({"starting": str(root), "step": name}), flush=True)
    with (root / (name + ".log")).open("w") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    records.append({"command": command, "seconds": time.monotonic() - start, "exit_code": proc.returncode})
    (root / "build-record.json").write_text(json.dumps(records, indent=2) + "\n")
    proc.check_returncode()
print(json.dumps({"built": str(build)}), flush=True)

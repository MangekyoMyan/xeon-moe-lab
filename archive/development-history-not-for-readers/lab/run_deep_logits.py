#!/usr/bin/env python3
"""Compare full-vocabulary logits for a fixed 127-token prefix and 33 continuation tokens."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("variants", nargs="+", choices=["baseline", "moe", "q8-rows4", "numa", "combined", "combined-numa"])
args = parser.parse_args()
if "baseline" not in args.variants or len(args.variants) < 2:
    parser.error("Include baseline and at least one candidate")
lab = Path(__file__).resolve().parent
out = lab / "results" / ("deep-logits-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
out.mkdir()
model = lab / "models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf"
records = []
for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
    if os.environ.get(key): raise SystemExit("Unset " + key)
for variant in args.variants:
    root = lab / ("review-deep-pools-20261019/baseline" if variant == "baseline" else
                  "review-deep-" + variant.removesuffix("-numa") + "-20261019" if variant == "combined-numa" else
                  "review-deep-" + variant + "-20261019")
    binary = root / "deepseek-logits-probe"
    compile_command = ["c++", "-std=c++17", "-O2", "-I", str(root / "source/include"),
        "-I", str(root / "source/ggml/include"), str(lab / "deepseek-logits-probe.cpp"),
        "-L", str(root / "build/bin"), "-Wl,-rpath," + str(root / "build/bin"),
        "-lllama", "-lggml-cpu", "-lggml-base", "-pthread", "-o", str(binary)]
    with (out / (variant + ".compile.log")).open("w") as log:
        subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    output = out / (variant + ".bin")
    command = ["numactl", "--physcpubind=0-23", "--interleave=0,1", str(binary), str(model), str(output)]
    env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24", OMP_MAX_ACTIVE_LEVELS="1",
               GGML_CPU_NUMA_REPLICA="1" if variant in ("numa", "combined-numa") else "0",
               GGML_CPU_MOE_PAIR_TRACE="0", GGML_CPU_NUMA_REPLICA_TRACE="0")
    print(json.dumps({"starting": variant}), flush=True)
    start = time.monotonic()
    with (out / (variant + ".stdout")).open("w") as stdout, (out / (variant + ".stderr")).open("w") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, timeout=900)
    row = {"variant": variant, "command": command, "compile_command": compile_command,
           "exit_code": proc.returncode, "wall_seconds_diagnostic_only": time.monotonic() - start,
           "cpu_library_sha256": hashlib.sha256((root / "build/bin/libggml-cpu.so").read_bytes()).hexdigest()}
    if proc.returncode == 0:
        raw = output.read_bytes()
        vocabulary, count, prefix = struct.unpack_from("<iii", raw)
        row.update({"output_sha256": hashlib.sha256(raw).hexdigest(), "vocabulary": vocabulary,
                    "records": count, "prefix_tokens": prefix, "logits_compared": vocabulary * count,
                    "bytes": len(raw)})
        assert len(raw) == 12 + 160 * 4 + count * (8 + vocabulary * 4)
    records.append(row)
    (out / "comparison.json").write_text(json.dumps({"runs": records, "diagnostic_only": True}, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    proc.check_returncode()
equal = len({r["output_sha256"] for r in records}) == 1
(out / "comparison.json").write_text(json.dumps({"runs": records, "all_outputs_bit_identical": equal,
    "diagnostic_only": True, "scope": "Full vocabulary at positions 126-159, fixed identical input tokens; separate natural-language checks required"}, indent=2) + "\n")
print(json.dumps({"saved": str(out / "comparison.json"), "all_outputs_bit_identical": equal}), flush=True)
if not equal: raise SystemExit("Full-model logits differ; inspect before adoption")

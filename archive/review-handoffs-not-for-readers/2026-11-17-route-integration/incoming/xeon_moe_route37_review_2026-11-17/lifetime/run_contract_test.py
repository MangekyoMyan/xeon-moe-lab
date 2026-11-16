#!/usr/bin/env python3
"""Build only the unchanged route37 collector and a local contract harness.

No supplied build script, ggml graph, model, GPU, or live API is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--compiler", default="g++")
p.add_argument("--disable-leak-check", action="store_true", help="ASan/UBSan still active; LSan cannot run under this workspace's ptrace")
args = p.parse_args()
repo = args.repo.resolve()
out = Path(__file__).resolve().parent
route = repo / "archive/review-handoffs-not-for-readers/2026-11-17-route-observation/route37"
include = repo / "archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/deepseek41-bringup-20261101/source/ggml/include"
source = route / "route37.cpp"
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == "8f8aca9be203043ea5cb5f82bfc90cdcfcd37e8ed027faaccaf6e7f5b4ec8865", "unexpected collector source"
cmd = [args.compiler, "-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
       "-pthread", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
       "-I", str(route), "-I", str(include), str(source),
       str(out / "collector_contract_test.cpp"), "-o", str(out / "collector_contract_test")]
build = subprocess.run(cmd, capture_output=True, text=True)
(out / "compile.stdout.txt").write_text(build.stdout)
(out / "compile.stderr.txt").write_text(build.stderr)
run = None
run_env = os.environ.copy()
if args.disable_leak_check:
    run_env["ASAN_OPTIONS"] = "detect_leaks=0"
if build.returncode == 0:
    run = subprocess.run([str(out / "collector_contract_test")], capture_output=True, text=True, env=run_env)
    (out / "contract_results.jsonl").write_text(run.stdout)
    (out / "run.stderr.txt").write_text(run.stderr)
receipt = {
    "scope": "unchanged collector plus host-buffer stub and direct hooks; no ggml graph or model",
    "source_sha256_before": source_sha,
    "source_sha256_after": hashlib.sha256(source.read_bytes()).hexdigest(),
    "compile_command": cmd,
    "compile_exit": build.returncode,
    "run_exit": None if run is None else run.returncode,
    "leak_check_disabled": args.disable_leak_check,
    "test_source_sha256": hashlib.sha256((out / "collector_contract_test.cpp").read_bytes()).hexdigest(),
    "result_sha256": None if run is None else hashlib.sha256(run.stdout.encode()).hexdigest(),
}
(out / "contract_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
if build.stdout:
    print(build.stdout)
if build.stderr:
    print(build.stderr)
if run is not None:
    print(run.stdout)
    if run.stderr:
        print(run.stderr)
raise SystemExit(build.returncode if run is None else run.returncode)

#!/usr/bin/env python3
"""Rebuild the archived CPU sources into an isolated engine tree."""
import argparse
import hashlib
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("accepted", type=pathlib.Path, help="repository with scripts/prepare_source.py")
p.add_argument("git_repo", type=pathlib.Path, help="checkout containing the fixed archive commit")
p.add_argument("destination", type=pathlib.Path)
p.add_argument("variant", choices=["current", "08", "09"])
args = p.parse_args()
src = args.destination.resolve()
description = json.loads((HERE / "inputs.json").read_text())
subprocess.run(["python3", args.accepted / "scripts/prepare_source.py", "--destination", src], check=True)
for item in description["archived_files"]:
    blob = subprocess.check_output(["git", "-C", args.git_repo, "show", description["commit"] + ":" + item["repository_path"]])
    if hashlib.sha256(blob).hexdigest() != item["sha256"]:
        raise SystemExit("Unexpected source hash for " + item["target"])
    (src / item["target"]).write_bytes(blob)
quants = src / "ggml/src/ggml-cpu/arch/x86/quants.c"
if hashlib.sha256(quants.read_bytes()).hexdigest() != description["x86_quants_sha256"]:
    raise SystemExit("Unexpected x86 quants.c")
with (src / "CMakeLists.txt").open("a") as f:
    f.write("""
target_sources(ggml-cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/mx-rhs.cpp")
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/mx-rhs.cpp"
    TARGET_DIRECTORY ggml-cpu APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
""")
patches = []
if args.variant in ("08", "09"):
    patches.append("08-current-mixed.patch")
if args.variant == "09":
    patches.append("09-selected-input-only.patch")
for patch in patches:
    subprocess.run(["git", "apply", "--check", HERE / patch], cwd=src, check=True)
    subprocess.run(["git", "apply", HERE / patch], cwd=src, check=True)

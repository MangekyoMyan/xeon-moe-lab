#!/usr/bin/env python3
"""Check the canonical rebuild against the isolated candidate's source and outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

lab = Path(__file__).resolve().parent
source, build = lab / "llama-private", lab / "build-avx-deep"
candidate = lab / "review-deep-combined-20261019"
output = lab / "results/deep-review-rebuild-check"
output.mkdir()
manifest = json.loads((candidate / "prepare.json").read_text())
for relative, expected in manifest["files"].items():
    if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected["after"]:
        raise SystemExit("Unexpected adopted source: " + relative)

def cpu_flags(root, src):
    entries = json.loads((root / "compile_commands.json").read_text())
    return {str(Path(e["file"]).relative_to(src)): e["command"].replace(str(root), "BUILD").replace(str(src), "SOURCE")
            for e in entries if "/ggml/src/ggml-cpu/" in e["file"]}

a, b = cpu_flags(candidate / "build", candidate / "source"), cpu_flags(build, source)
if a != b:
    (output / "compile-command-difference.json").write_text(json.dumps({"candidate": a, "rebuilt": b}, indent=2) + "\n")
    raise SystemExit("Review CPU compiler differences before proceeding")
env = dict(os.environ, OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="24", OMP_MAX_ACTIVE_LEVELS="1",
           GGML_CPU_NUMA_REPLICA="1", GGML_CPU_MOE_PAIR_TRACE="1", GGML_CPU_NUMA_REPLICA_TRACE="1")
with (output / "quantize.log").open("w") as log:
    subprocess.run([str(build / "bin/test-quantize-fns")], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
references = {r["name"]: r["reference_sha256"] for r in json.loads((candidate / "unit-validation.json").read_text())}
records = []
for name, file in (("moe-pair", "moe-pair-probe.cpp"), ("kernel-graph", "deep-kernel-graph-probe.cpp"),
                   ("numa-replica", "numa-replica-probe.cpp")):
    binary = output / (name + "-probe")
    command = ["c++", "-std=c++17", "-O2", "-I", str(source / "ggml/include"),
        "-I", str(source / "ggml/src"), "-I", str(source / "ggml/src/ggml-cpu"), str(lab / file),
        "-L", str(build / "bin"), "-Wl,-rpath," + str(build / "bin"),
        "-lggml", "-lggml-cpu", "-lggml-base", "-pthread", "-o", str(binary)]
    with (output / (name + ".compile.log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = output / (name + ".bin")
    with (output / (name + ".stdout")).open("w") as stdout, (output / (name + ".stderr")).open("w") as stderr:
        subprocess.run([str(binary), str(raw)], stdout=stdout, stderr=stderr, env=env, check=True)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    records.append({"name": name, "compile_command": command, "sha256": digest,
                    "reference_sha256": references[name], "bit_identical": digest == references[name]})
report = {"source_hashes_match": True, "normalized_cpu_compile_commands_match": True,
          "quantize_test_passed": True, "probes": records, "all_passed": all(r["bit_identical"] for r in records),
          "artifacts": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (build / "bin").iterdir()
                        if p.is_file() and not p.is_symlink()}}
(output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["all_passed"]:
    raise SystemExit("Rebuilt output mismatch")

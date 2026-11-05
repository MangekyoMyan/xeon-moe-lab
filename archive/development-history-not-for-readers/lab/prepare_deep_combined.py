#!/usr/bin/env python3
"""Combine measured candidates in an isolated tree; leave the active launcher unchanged."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

lab = Path(__file__).resolve().parent
source = lab / "review-deep-pools-20261019/baseline/source"
root = lab / "review-deep-combined-20261019"
if root.exists(): raise SystemExit("Combined experiment already exists")
root.mkdir()
target = root / "source"
shutil.copytree(source, target)
patches = [lab / "review-deep-moe-20261019/moe-pair.patch",
           lab / "review-deep-q8-rows4-20261019/q8-rows4-k8192.patch",
           lab / "review-deep-numa-20261019/numa-replica.patch"]
for patch in patches:
    with patch.open("rb") as handle:
        subprocess.run(["patch", "-p1", "--batch", "--fuzz=0", "--no-backup-if-mismatch"],
                       stdin=handle, cwd=target, check=True)
cpu = Path("ggml/src/ggml-cpu")
p = target / cpu / "ggml-cpu.c"
s = p.read_text()
old = "if (type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 && ne11 == 1 && ne00 == 8192)"
if s.count(old) != 1: raise RuntimeError("Q8 dispatch anchor changed")
s = s.replace(old, "if (!params->use_ref && type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 && ne11 == 1 && ne00 == 8192)")
p.write_text(s)
p = target / cpu / "numa-replica.cpp"
s = p.read_text()
old = '    const char * env = std::getenv("GGML_CPU_NUMA_REPLICA");\n    if (!env || std::strcmp(env, "1") != 0) return nullptr;'
new = '''    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CPU_NUMA_REPLICA");
        return env && std::strcmp(env, "1") == 0;
    }();
    if (!enabled) return nullptr;'''
if s.count(old) != 1: raise RuntimeError("NUMA configuration anchor changed")
p.write_text(s.replace(old, new))
files = [cpu / f for f in ("ggml-cpu.c", "ggml-cpu.cpp", "quants.h", "arch/x86/quants.c",
                           "CMakeLists.txt", "numa-replica.cpp", "numa-replica.h")]
records = {}
patch = []
for rel in files:
    before = (source / rel).read_text() if (source / rel).exists() else ""
    after = (target / rel).read_text()
    patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile="a/" + str(rel) if before else "/dev/null", tofile="b/" + str(rel)))
    records[str(rel)] = {"before": hashlib.sha256(before.encode()).hexdigest() if before else None,
                        "after": hashlib.sha256(after.encode()).hexdigest()}
(root / "combined.patch").write_text("".join(patch))
(root / "prepare.json").write_text(json.dumps({"source": str(source), "input_patches": [str(p) for p in patches],
    "files": records, "additional_guards": ["Respect reference mode in Q8 dispatch", "Read NUMA opt-in once"],
    "adopted": False}, indent=2) + "\n")

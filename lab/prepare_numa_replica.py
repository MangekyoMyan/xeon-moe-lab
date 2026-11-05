#!/usr/bin/env python3
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
if output.exists() or output.is_relative_to(source):
    parser.error("Use a new output outside the source directory")
output.mkdir(parents=True)
target = output / "source"
shutil.copytree(source, target)
cpu = Path("ggml/src/ggml-cpu")
changes = {}
for rel, old, new in [
    (cpu / "CMakeLists.txt", "        ggml-cpu/traits.cpp\n", "        ggml-cpu/numa-replica.cpp\n        ggml-cpu/numa-replica.h\n        ggml-cpu/traits.cpp\n"),
    (cpu / "ggml-cpu.cpp", "#include \"ggml-cpu.h\"", "#include \"ggml-cpu.h\"\n#include \"numa-replica.h\""),
]:
    before = (target / rel).read_text()
    if before.count(old) != 1: raise ValueError(str(rel))
    after = before.replace(old, new, 1)
    if rel.name == "ggml-cpu.cpp":
        old = "        std::vector<ggml_backend_buffer_type_t> bufts;\n"
        if after.count(old) != 1: raise ValueError("extra buffer registration")
        after = after.replace(old, old + "\n        if (auto * replica = ggml_backend_cpu_numa_replica_buffer_type()) {\n            bufts.push_back(replica);\n        }\n", 1)
    changes[rel] = before, after
changes[cpu / "numa-replica.cpp"] = "", (Path(__file__).parent / "numa-replica.cpp").read_text()
changes[cpu / "numa-replica.h"] = "", '#pragma once\n#include "ggml-backend.h"\n\nggml_backend_buffer_type_t ggml_backend_cpu_numa_replica_buffer_type();\n'
patch = []
hashes = {}
for rel, (before, after) in changes.items():
    (target / rel).write_text(after)
    patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                    fromfile="a/" + str(rel) if before else "/dev/null", tofile="b/" + str(rel)))
    hashes[str(rel)] = {"before": hashlib.sha256(before.encode()).hexdigest() if before else None,
                        "after": hashlib.sha256(after.encode()).hexdigest()}
(output / "numa-replica.patch").write_text("".join(patch))
(output / "prepare.json").write_text(json.dumps({"source": str(source), "files": hashes,
    "opt_in_environment": "GGML_CPU_NUMA_REPLICA=1", "adopted": False}, indent=2) + "\n")

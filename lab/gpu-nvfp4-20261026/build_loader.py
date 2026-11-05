#!/usr/bin/env python3
"""Recompile one loader object and link an isolated, ABI-identical runtime."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
base = root.parent / "v100-prep-20261024"
build = base / "build-cuda"
runtime = root / "runtime"
runtime.mkdir(exist_ok=True)
flags = {}
for line in (build / "src/CMakeFiles/llama.dir/flags.make").read_text().splitlines():
    if " = " in line:
        key, value = line.split(" = ", 1)
        flags[key] = shlex.split(value)
obj = root / "patch/llama-model-loader.cpp.o"
compile_cmd = ["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
               *flags["CXX_FLAGS"], "-c", str(root / "patch/llama-model-loader.cpp"), "-o", str(obj)]
link_cmd = shlex.split((build / "src/CMakeFiles/llama.dir/link.txt").read_text())
link_cmd = [str(obj) if token == "CMakeFiles/llama.dir/llama-model-loader.cpp.o" else token for token in link_cmd]
link_cmd[link_cmd.index("-o") + 1] = str(runtime / "libllama.so.0.5.0")
link_cmd = ["-Wl,--dependency-file=" + str(root / "patch/link.d") if token.startswith("-Wl,--dependency-file=") else token for token in link_cmd]
record = {"compile": compile_cmd, "link": link_cmd, "cwd": str(build / "src")}
with (root / "loader-build.log").open("w") as log:
    subprocess.run(compile_cmd, cwd=build / "src", stdout=log, stderr=log, check=True)
    subprocess.run(link_cmd, cwd=build / "src", stdout=log, stderr=log, check=True)
for name in ["libllama.so", "libllama.so.0"]:
    link = runtime / name
    if link.is_symlink():
        assert link.readlink() == Path("libllama.so.0.5.0")
    else:
        link.symlink_to("libllama.so.0.5.0")
record["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [root / "patch/llama-model-loader.cpp", runtime / "libllama.so.0.5.0", base / "source/src/llama-model-loader.cpp"]}
(root / "loader-build.json").write_text(json.dumps(record, indent=2) + "\n")
print("Isolated loader linked successfully", flush=True)

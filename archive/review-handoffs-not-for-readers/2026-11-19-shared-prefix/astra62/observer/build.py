#!/usr/bin/env python3
"""Build the host-only diagnostic interposer against the installed CUPTI."""
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
lab = here.parents[2]
src = here.parent / "trace169.cpp"
cuda = lab / "v100-prep-20261024/cuda-12.9.2"
cupti = lab / "strata-tc-study-20261031/cupti/cuda_cupti-linux-x86_64-12.9.79-archive"
ggml = lab / "deepseek41-bringup-20261101/source/ggml/include"
output = here / "libtrace169.so"
compiler = "/usr/bin/g++"
depfile = here / "headers.d"
cmd = [compiler, "-std=c++17", "-O2", "-g", "-march=ivybridge", "-mtune=ivybridge",
       "-fPIC", "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror", "-pthread",
       "-I" + str(ggml), "-I" + str(cupti/"include"), "-I" + str(cuda/"include"),
       "-MMD", "-MF", str(depfile), "-shared", str(src), "-Wl,-z,defs",
       "-L" + str(cupti/"lib"), "-Wl,-rpath," + str(cupti/"lib"), "-lcupti", "-ldl", "-o", str(output)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

with (here / "build.stdout.log").open("wb") as out, (here / "build.stderr.log").open("wb") as err:
    run = subprocess.run(cmd, cwd=here, stdout=out, stderr=err)
sources = [src, Path(__file__).resolve()]
if depfile.exists():
    deptext = depfile.read_text().split(":", maxsplit=1)[1].replace("\\\n", " ")
    sources.extend(Path(n) for n in deptext.split())
receipt = dict(command=cmd, cwd=str(here), compiler=compiler, compiler_sha256=sha(compiler),
               source_sha256={str(p): sha(p) for p in sorted(set(sources))},
               returncode=run.returncode, library=str(output),
               cupti_path=str((cupti/"lib/libcupti.so.12").resolve()),
               cupti_sha256=sha(cupti/"lib/libcupti.so.12"))
if run.returncode == 0:
    receipt["library_sha256"] = sha(output)
    receipt["dynamic_section"] = subprocess.check_output(["readelf", "-dW", str(output)], text=True)
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(output)], text=True)
    receipt["trace_exports"] = [s.split()[-1] for s in symbols.splitlines()
                                if " s169_" in s or " ggml_backend_" in s]
    required = {"ggml_backend_tensor_copy", "ggml_backend_graph_compute_async", "s169_trace_initialize",
                "s169_trace_dump", "s169_trace_call_begin", "s169_trace_call_end",
                "s169_trace_window_begin", "s169_trace_window_end"}
    assert required == set(receipt["trace_exports"])
(here / "BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(run.returncode)

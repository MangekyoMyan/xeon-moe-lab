#!/usr/bin/env python3
"""Build the small Q6 scale/prefetch candidate in an isolated source tree."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

LAB = Path(__file__).resolve().parent
BASE = LAB / "review-full-replica-20261020"
ROOT = LAB / "review-q6-scales-20261020"
ROOT.mkdir(exist_ok=False)
shutil.copytree(BASE / "source", ROOT / "source")
relative = "ggml/src/ggml-cpu/arch/x86/quants.c"
candidate = LAB / "q6-xeon-v2-20261020/scales16-pf2.c"
shutil.copyfile(candidate, ROOT / "source" / relative)
cpu_relative = "ggml/src/ggml-cpu/ggml-cpu.c"
cpu_path = ROOT / "source" / cpu_relative
cpu_text = cpu_path.read_text()
anchor = "    // Now select a reasonable chunk size.\n"
assert cpu_text.count(anchor) == 1
static_rows = """#if defined(__AVX__) && !defined(__AVX2__)
    if (!params->use_ref && src0->type == GGML_TYPE_Q6_K && nr1 == 1) {
        const char * env = getenv("GGML_CPU_Q6_STATIC_ROWS");
        if (env && strcmp(env, "1") == 0) {
            const int64_t rows = nr0 / nth;
            const int64_t tail = nr0 % nth;
            const int64_t begin = rows * ith + MIN(ith, tail);
            const int64_t end = begin + rows + (ith < tail);
            ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, 1, begin, end, 0, 1);
            return;
        }
    }
#endif

"""
cpu_path.write_text(cpu_text.replace(anchor, static_rows + anchor))
configure = [
    "cmake", "-S", str(ROOT / "source"), "-B", str(ROOT / "build"),
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DGGML_NATIVE=ON",
    "-DGGML_OPENMP=ON", "-DGGML_BLAS=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
    "-DGGML_OPENCL=OFF", "-DGGML_METAL=OFF", "-DGGML_SYCL=OFF", "-DLLAMA_BUILD_UI=OFF",
    "-DLLAMA_BUILD_MTMD=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_BUILD_COMMON=ON", "-DLLAMA_BUILD_TOOLS=ON",
]
build = ["cmake", "--build", str(ROOT / "build"), "--target", "llama-bench",
         "llama-completion", "test-backend-ops", "test-quantize-fns", "-j", "8"]
manifest = {"changed_files": [relative, cpu_relative], "commands": [configure, build],
            "before_sha256": hashlib.sha256((BASE / "source" / relative).read_bytes()).hexdigest(),
            "after_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "cpu_before_sha256": hashlib.sha256(cpu_text.encode()).hexdigest(),
            "cpu_after_sha256": hashlib.sha256(cpu_path.read_bytes()).hexdigest()}
(ROOT / "prepare.json").write_text(json.dumps(manifest, indent=2) + "\n")
for label, command in (("configure", configure), ("build", build)):
    print(label, flush=True)
    with (ROOT / (label + ".log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
print("Isolated build completed", flush=True)

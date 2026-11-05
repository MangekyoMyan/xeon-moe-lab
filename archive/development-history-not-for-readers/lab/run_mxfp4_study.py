#!/usr/bin/env python3
"""Compare existing AVX quantized dot kernels without changing the runtime."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import run_quantize_tradeoff as runner

LAB = Path(__file__).resolve().parent
OUT = LAB / "results/mxfp4-study-20261022"
BIN = LAB / "mxfp4-study-20261022"
SOURCE = LAB / "review-glm-q8-columns4-20261020/source"
LIB = LAB / "review-glm-q8-columns4-20261020/build/bin"
TYPES = ["q8_0", "q4_K", "q4_0", "iq4_nl", "mxfp4"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    prior = json.loads((LAB / "results/quantize-tradeoff-20261021/before.json").read_text())
    files = list(prior["sha256"]) + [str(LIB / "libggml-base.so")]
    return {"settings": {p: Path(p).read_text().strip() for p in prior["settings"]},
            "sha256": {p: sha(p) for p in files}}


def prepare():
    OUT.mkdir(exist_ok=False)
    BIN.mkdir(exist_ok=False)
    (OUT / "before.json").write_text(json.dumps(audit(), indent=2) + "\n")
    for name, args in (("cpu", ["lscpu"]), ("numa", ["numactl", "--hardware"])):
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        (OUT / ("host-" + name + ".txt")).write_text(result.stdout)
    original = LAB / "quantize-tradeoff-20261021"
    for name in ("test-quantize-perf-fixed", "test-quantize-perf-fixed.cpp", "test-quantize-perf-rhs.patch"):
        shutil.copy2(original / name, BIN / name)
    code = (original / "quantize-matvec.cpp").read_text()
    start = code.index("    const ggml_type types[] =")
    end = code.index("    const int patterns", start)
    code = code[:start] + "    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0,\n        GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4};\n" + code[end:]
    (BIN / "quantize-matvec.cpp").write_text(code)
    command = ["c++", "-O3", "-march=ivybridge", "-std=c++17", "-fopenmp",
               "-I" + str(SOURCE / "ggml/include"), "-I" + str(SOURCE / "ggml/src"),
               str(BIN / "quantize-matvec.cpp"), "-L" + str(LIB), "-Wl,-rpath," + str(LIB),
               "-lggml-cpu", "-lggml-base", "-o", str(BIN / "quantize-matvec")]
    result = subprocess.run(command, capture_output=True, text=True)
    (BIN / "build.json").write_text(json.dumps({"command": command, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr}, indent=2) + "\n")
    result.check_returncode()
    paths = [p for p in BIN.iterdir() if p.is_file()]
    paths += [LIB / "libggml-cpu.so", LIB / "libggml-base.so",
              SOURCE / "ggml/src/ggml-cpu/arch/x86/quants.c",
              SOURCE / "ggml/src/ggml-cpu/ggml-cpu.c", SOURCE / "ggml/src/ggml-quants.c"]
    (BIN / "hashes.json").write_text(json.dumps({str(p): sha(p) for p in paths}, indent=2) + "\n")
    code = (SOURCE / "ggml/src/ggml-cpu/arch/x86/quants.c").read_text()
    begin = code.index("void ggml_vec_dot_mxfp4_q8_0(")
    end = code.index("void ggml_vec_dot_nvfp4_q8_0(", begin)
    (BIN / "runtime-mxfp4-kernel.txt").write_text(code[begin:end])
    (OUT / "WORK_STATE.txt").write_text("Prepared serial CPU-only MXFP4 comparisons. Runtime unchanged.\n")


def main():
    with (LAB / "results/glm-air-q8-20261020/.experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        runner.OUT = OUT
        one = ["numactl", "--physcpubind=0", "--membind=0"]
        dual = ["numactl", "--physcpubind=0-23", "--interleave=0,1"]
        runner.run("validation", dual + [str(BIN / "quantize-matvec"), "4096", "24", "24", "0.05", "1"])
        for repeat in range(2):
            for regime, size, count in (("cache", "4096", "500000"), ("stream", "33554432", "20")):
                types = TYPES if repeat % 2 == 0 else list(reversed(TYPES))
                args = [arg for kind in types for arg in ("--type", kind)]
                runner.run(f"quant-perf-{regime}-{repeat}", one + [str(BIN / "test-quantize-perf-fixed"),
                    *args, "--op", "vec_dot_q", "--op", "dequantize_row_q", "--size", size, "-i", count])
        for threads in (1, 12, 24):
            for regime, rows in (("cache", 24), ("stream", 65536)):
                runner.run(f"matvec-k4096-{regime}-t{threads}", dual + [str(BIN / "quantize-matvec"),
                    "4096", str(rows), str(threads), "0.5", "6"])
        for k, rows in ((1408, 196608), (2880, 98304), (16384, 16384)):
            runner.run(f"matvec-k{k}-stream-t24", dual + [str(BIN / "quantize-matvec"),
                str(k), str(rows), "24", "0.5", "6"])
        (OUT / "complete.json").write_text(json.dumps({"complete": True}) + "\n")
        after = audit()
        before = json.loads((OUT / "before.json").read_text())
        assert after == before
        (OUT / "after-measurements.json").write_text(json.dumps(after, indent=2) + "\n")
        (OUT / "WORK_STATE.txt").write_text("Kernel measurements completed. Hardware counter observations and report pending.\n")
        print("MXFP4 kernel measurements complete", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate a private Q8 row-sharing build against the unchanged B runtime."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from glm_q8_experiment import LAB, OUT, MODEL, BUILDS, environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()
    root = args.candidate.resolve()
    target = OUT / ("validation-" + root.name)
    with (OUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target.mkdir(exist_ok=False)
        env = environment()
        report = {"complete": False, "checks": [], "logits": []}

        def save():
            (target / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

        def execute(label, command, timeout=900):
            print("Validation:", label, flush=True)
            with (target / (label + ".stdout")).open("w") as stdout, (target / (label + ".stderr")).open("w") as stderr:
                subprocess.run(command, stdout=stdout, stderr=stderr, env=env,
                               check=True, timeout=timeout)
            report["checks"].append({"label": label, "command": command, "passed": True})
            save()

        if "columns4" in root.name:
            guard = target / "column-check"
            execute("compile-column-check", [
                "cc", "-O2", "-march=native",
                "-I" + str(root / "source/ggml/include"), "-I" + str(root / "source/ggml/src"),
                str(LAB / "glm-q8-column-check.c"), "-L" + str(root / "build/bin"),
                "-Wl,-rpath," + str(root / "build/bin"), "-lggml-cpu", "-lggml-base",
                "-ldl", "-lm", "-o", str(guard)])
            execute("column-check", [str(guard), str(root / "build/bin/libggml-cpu.so")])
        for name, options in (
            ("test-quantize-fns", []),
            ("test-backend-ops", ["-b", "CPU", "-o", "MUL_MAT,MUL_MAT_ID", "-p", "type_a=q8_0"]),
        ):
            execute(name, ["numactl", "--physcpubind=0-23", "--interleave=0,1",
                           str(root / "build/bin" / name), *options])
        counts = re.findall(r"(\d+)/(\d+) tests passed",
                            (target / "test-backend-ops.stdout").read_text())
        assert counts and all(int(a) == int(b) and int(b) > 0 for a, b in counts)
        report["backend_test_counts"] = counts
        dense = (LAB / "deep-kernel-graph-probe.cpp").read_text()
        dense = dense.replace("{GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XXS}", "{GGML_TYPE_Q8_0}")
        dense = dense.replace("{1, 4}", "{1, 4, 12, 24}")
        dense = dense.replace("{8192, 65", "{4096, 65").replace("{8192, 35", "{1408, 35")
        dense = dense.replace("{unit,   17, 3", "{4096, 17, 3")
        dense = dense.replace("{unit,   35", "{1408, 35")
        (target / "dense.cpp").write_text(dense)
        moe = (LAB / "moe-pair-probe.cpp").read_text()
        moe = moe.replace("{GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_0}", "{GGML_TYPE_Q8_0}")
        moe = moe.replace("{1, 4, 12}", "{1, 4, 12, 24}")
        moe = moe.replace("bool row_view, separate_input, duplicate_ids;",
                          "bool row_view, separate_input, duplicate_ids, indexed_input;")
        start = moe.index("    const shape shapes[] = {")
        end = moe.index("\n    int cases = 0;", start)
        moe = moe[:start] + """    const shape shapes[] = {
        {4096, 1, 3, 1, 1, false, false, false, false},
        {4096, 3, 9, 6, 1, false, false, false, false},
        {4096, 65, 128, 8, 1, false, false, false, false},
        {4096, 1408, 8, 8, 1, false, false, false, false},
        {1408, 4096, 8, 8, 1, false, false, false, true},
        {1408, 67, 9, 8, 1, false, false, true, true},
        {4096, 35, 5, 2, 2, false, false, false, false},
        {1408, 35, 5, 2, 1, true, false, false, true},
        {4096, 17, 3, 2, 1, false, true, false, false},
        {4096, 65, 8, 8, 8, false, false, false, false},
        {1408, 67, 8, 8, 9, false, false, false, true},
        {4096, 35, 5, 2, 32, true, false, false, false},
        {1408, 35, 5, 2, 9, true, false, false, true},
        {4096, 17, 3, 2, 8, false, true, true, false},
    };""" + moe[end:]
        moe = moe.replace("GGML_TYPE_F32, s.k, 1, s.tokens",
                          "GGML_TYPE_F32, s.k, s.indexed_input ? s.selected : 1, s.tokens")
        (target / "moe.cpp").write_text(moe)
        logits = (LAB / "deepseek-logits-probe.cpp").read_text()
        logits = logits.replace("model_params.load_mode = LLAMA_LOAD_MODE_MMAP;",
                                "model_params.load_mode = LLAMA_LOAD_MODE_MMAP;\n"
                                "    model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;\n"
                                "    model_params.use_extra_bufts = false;")
        logits = logits.replace("context_params.n_threads = 12;", "context_params.n_threads = 24;")
        logits = logits.replace("ggml_threadpool_params_default(12)", "ggml_threadpool_params_default(24)")
        logits = logits.replace("n_batch = 512", "n_batch = 128").replace("n_ubatch = 512", "n_ubatch = 128")
        (target / "logits.cpp").write_text(logits)
        for label, source, build in (
            ("B", LAB / "llama-private-prefetch", BUILDS["B"]),
            ("C", root / "source", root / "build"),
        ):
            for name in ("dense", "moe", "logits"):
                binary = target / f"{label}-{name}"
                command = ["c++", "-std=c++17", "-O2",
                           "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
                           str(target / (name + ".cpp")), "-L" + str(build / "bin"),
                           "-Wl,-rpath," + str(build / "bin"), "-lllama", "-lggml",
                           "-lggml-cpu", "-lggml-base", "-pthread", "-o", str(binary)]
                execute(f"compile-{label}-{name}", command)
                output = target / f"{label}-{name}.bin"
                command = ["numactl", "--physcpubind=0-23", "--interleave=0,1", str(binary)]
                if name == "logits":
                    command.append(str(MODEL))
                command.append(str(output))
                execute(f"{label}-{name}", command)
                raw = output.read_bytes()
                row = {"label": label, "test": name, "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}
                if name == "logits":
                    vocab, records, prefix = struct.unpack_from("<iii", raw)
                    assert (vocab, records, prefix) == (151552, 34, 127)
                    assert len(raw) == 12 + 160 * 4 + records * (8 + vocab * 4)
                    row["logits_compared"] = vocab * records
                report["logits"].append(row)
                save()
        report["bit_identical"] = all(
            len({r["sha256"] for r in report["logits"] if r["test"] == test}) == 1
            for test in ("dense", "moe", "logits"))
        assert report["bit_identical"]
        report["complete"] = True
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

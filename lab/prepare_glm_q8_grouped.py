#!/usr/bin/env python3
"""Prepare a private row-sharing candidate after microbenchmark selection."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

LAB = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, choices=(2, 4), required=True)
    ap.add_argument("--prefetch", type=int, choices=(0, 8), required=True)
    args = ap.parse_args()
    rows, distance = args.rows, args.prefetch
    root = LAB / f"review-glm-q8-rows{rows}-pf{distance}-20261020"
    base = LAB / "llama-private-prefetch"
    root.mkdir(exist_ok=False)
    shutil.copytree(base, root / "source")
    src = root / "source"
    relative = Path("ggml/src/ggml-cpu")
    qfile = src / relative / "arch/x86/quants.c"
    original = qfile.read_text()
    start = original.index("void ggml_vec_dot_q8_0_q8_0_rows4(")
    end = original.index("\n#endif", start)
    name = f"ggml_vec_dot_q8_0_q8_0_rows{rows}_pf{distance}"
    fn = original[start:end].replace("ggml_vec_dot_q8_0_q8_0_rows4", name)
    fn = fn.replace("[4]", f"[{rows}]").replace("r < 4", f"r < {rows}")
    if distance:
        anchor = "    for (; ib + 1 < nb; ib += 2) {\n"
        assert fn.count(anchor) == 1
        fn = fn.replace(anchor, anchor +
            f"        if (ib + {distance} < nb) {{\n"
            f"            for (int r = 0; r < {rows}; ++r) {{\n"
            f"                _mm_prefetch((const char *) &x[r][ib + {distance}], _MM_HINT_T0);\n"
            "            }\n"
            "        }\n")
    qfile.write_text(original[:end] + "\n\n" + fn + original[end:])
    header = src / relative / "quants.h"
    text = header.read_text()
    declaration = next(line for line in text.splitlines()
                       if line.startswith("void ggml_vec_dot_q8_0_q8_0_rows4("))
    text = text.replace(declaration, declaration + "\n" +
                        declaration.replace("ggml_vec_dot_q8_0_q8_0_rows4", name))
    header.write_text(text)
    cpu = src / relative / "ggml-cpu.c"
    text = cpu.read_text()
    # Preserve the existing 8192-wide kernel and add only the measured widths.
    anchor = """                for (; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {"""
    assert text.count(anchor) == 1
    grouped = f"""#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
                if (!params->use_ref && type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 && ne11 == 1 && (ne00 == 4096 || ne00 == 1408)) {{
                    for (; ir0 + {rows - 1} < iir0 + blck_0 && ir0 + {rows - 1} < ir0_end; ir0 += {rows}) {{
                        {name}(ne00, &tmp[ir0 - iir0], src0_row + ir0 * nb01, nb01, src1_col);
                    }}
                }}
#endif
"""
    text = text.replace(anchor, grouped + anchor)
    anchor = "static void ggml_compute_forward_mul_mat_id_one_chunk(\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "    const bool use_ref,\n")
    for target in ("dst, src0, src1, ids, cur_a,", "pair, pair_weight, src1, ids, cur_a,"):
        assert text.count(target) == 1
        text = text.replace(target, "params->use_ref, " + target)
    anchor = """                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }"""
    assert text.count(anchor) == 1
    replacement = f"""                int64_t ir0 = iir0;
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
                if (!use_ref && type == GGML_TYPE_Q8_0 && ids->ne[1] == 1 && (ne00 == 4096 || ne00 == 1408)) {{
                    for (; ir0 + {rows - 1} < iir0 + blck_0 && ir0 + {rows - 1} < ir0_end; ir0 += {rows}) {{
                        {name}(ne00, &tmp[ir0 - iir0], src0_cur + ir0 * nb01, nb01, src1_col);
                    }}
                }}
#else
                GGML_UNUSED(use_ref);
#endif
                for (; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {{
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }}"""
    text = text.replace(anchor, replacement)
    cpu.write_text(text)
    changed = [relative / "arch/x86/quants.c", relative / "quants.h", relative / "ggml-cpu.c"]
    patch = []
    hashes = {}
    for rel in changed:
        before, after = (base / rel).read_bytes(), (src / rel).read_bytes()
        hashes[str(rel)] = {"before": hashlib.sha256(before).hexdigest(),
                            "after": hashlib.sha256(after).hexdigest()}
        patch.extend(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                         fromfile="a/" + str(rel), tofile="b/" + str(rel)))
    (root / "candidate.patch").write_text("".join(patch))
    configure = ["cmake", "-S", str(src), "-B", str(root / "build"),
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DGGML_NATIVE=ON", "-DGGML_OPENMP=ON", "-DGGML_BLAS=OFF",
                 "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_OPENCL=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_SYCL=OFF", "-DLLAMA_BUILD_UI=OFF",
                 "-DLLAMA_BUILD_MTMD=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TESTS=ON"]
    build = ["cmake", "--build", str(root / "build"), "--target", "llama-bench", "llama-completion",
             "test-backend-ops", "test-quantize-fns", "-j", "8"]
    (root / "prepare.json").write_text(json.dumps(
        {"arguments": vars(args), "hashes": hashes, "commands": [configure, build]}, indent=2) + "\n")
    for label, command in (("configure", configure), ("build", build)):
        print(label, root, flush=True)
        with (root / (label + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adapt the reviewed two-row patch to a measured four-row, K=8192 dispatch."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--patch", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
if output.exists() or output.is_relative_to(source):
    parser.error("Use a new output outside the source directory")
output.mkdir(parents=True)
target = output / "source"
shutil.copytree(source, target)
with args.patch.resolve().open("rb") as patch:
    subprocess.run(["patch", "-p1", "--fuzz=0", "--batch", "--no-backup-if-mismatch"],
                   cwd=target, stdin=patch, check=True)
paths = [Path("ggml/src/ggml-cpu") / x for x in ("arch/x86/quants.c", "quants.h", "ggml-cpu.c")]
patch_lines = []
hashes = {}
for rel in paths:
    original = (source / rel).read_text()
    text = (target / rel).read_text().replace("ggml_vec_dot_q8_0_q8_0_rows2", "ggml_vec_dot_q8_0_q8_0_rows4")
    if rel.name == "quants.c":
        start = text.index("void ggml_vec_dot_q8_0_q8_0_rows4(")
        end = text.index("\n#endif", start)
        fn = text[start:end]
        for old, new, count in [
            ("const block_q8_0 * x[2] = {vx, (const block_q8_0 *) ((const char *) vx + bx)};", "const block_q8_0 * x[4];", 1),
            ("__m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};", "__m256 acc[4];\n    for (int r = 0; r < 4; ++r) {\n        x[r] = (const block_q8_0 *) ((const char *) vx + r * bx);\n        acc[r] = _mm256_setzero_ps();\n    }", 1),
            ("r < 2", "r < 4", 2)]:
            if fn.count(old) != count:
                raise ValueError(f"Unexpected kernel source anchor {old}")
            fn = fn.replace(old, new)
        text = text[:start] + fn + text[end:]
    elif rel.name == "ggml-cpu.c":
        old = "type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 && ne11 == 1"
        if text.count(old) != 1:
            raise ValueError("Unexpected dispatch condition")
        text = text.replace(old, old + " && ne00 == 8192")
        old = "ir0 + 1 < iir0 + blck_0 && ir0 + 1 < ir0_end; ir0 += 2"
        if text.count(old) != 1:
            raise ValueError("Unexpected row loop")
        text = text.replace(old, "ir0 + 3 < iir0 + blck_0 && ir0 + 3 < ir0_end; ir0 += 4")
    (target / rel).write_text(text)
    patch_lines.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                          fromfile="a/" + str(rel), tofile="b/" + str(rel)))
    hashes[str(rel)] = {"before": hashlib.sha256(original.encode()).hexdigest(),
                        "after": hashlib.sha256(text.encode()).hexdigest()}
(output / "q8-rows4-k8192.patch").write_text("".join(patch_lines))
(output / "prepare.json").write_text(json.dumps({"source": str(source), "files": hashes, "adopted": False}, indent=2) + "\n")

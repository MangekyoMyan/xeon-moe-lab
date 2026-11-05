#!/usr/bin/env python3
"""Screen isolated AVX-only Q6 kernels with unchanged integer and float sums."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

LAB = Path(__file__).resolve().parent
BASE = LAB / "review-full-replica-20261020"
OUT = LAB / "q6-xeon-20261020"
SOURCE = BASE / "source/ggml/src/ggml-cpu/arch/x86/quants.c"
original = SOURCE.read_text()
start = original.index("void ggml_vec_dot_q6_K_q8_K(")
avx = original.index("#elif defined __AVX__", start)
end = original.index("\n#else", avx)
body = original[avx:end]


def streamed(vector_scale):
    prelude = body[:body.index("        int is = 0;")]
    lines = [prelude, "        for (int j = 0; j < 2; ++j) {\n"]
    if vector_scale:
        lines.append("            const __m128i scales16 = j == 0 ? scales_16_0 : scales_16_1;\n")
    for lane in (0, 1):
        lines.append("            {\n")
        lines.append(f"                const __m128i hi = _mm_loadu_si128((const __m128i *) (qh + j*32 + {lane*16}));\n")
        for group in range(4):
            pos = group * 32 + lane * 16
            off = (group % 2) * 32 + lane * 16
            lines.append("                {\n")
            lines.append(f"                    __m128i lo = _mm_loadu_si128((const __m128i *) (q4 + j*64 + {off}));\n")
            if group >= 2:
                lines.append("                    lo = _mm_srli_epi16(lo, 4);\n")
            lines.append(f"                    const __m128i high = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hi, {group*2}), m3), 4);\n")
            lines.append("                    const __m128i q = _mm_or_si128(_mm_and_si128(lo, m15), high);\n")
            lines.append(f"                    const __m128i v = _mm_loadu_si128((const __m128i *) (q8 + j*128 + {pos}));\n")
            if vector_scale:
                mask = 0x0100 + (group * 2 + lane) * 0x0202
                lines.append(f"                    const __m128i scale = _mm_shuffle_epi8(scales16, _mm_set1_epi16(0x{mask:04x}));\n")
            else:
                lines.append(f"                    const __m128i scale = _mm_set1_epi16(x[i].scales[j*8 + {group*2+lane}]);\n")
            lines.append(f"                    sumi_{lane} = _mm_add_epi32(sumi_{lane}, _mm_madd_epi16(scale, _mm_maddubs_epi16(q, v)));\n")
            lines.append("                }\n")
        lines.append("            }\n")
    lines.append("        }\n")
    lines.append(body[body.index("\n        sumi_0 = _mm_sub_epi32"):])
    return "".join(lines)


def prefetch(code, distance):
    needle = "    for (int i = 0; i < nb; ++i) {\n"
    assert code.count(needle) == 1
    hints = "".join(f"            _mm_prefetch((const char *) &x[i + {distance}] + {off}, _MM_HINT_T0);\n"
                    for off in (0, 64, 128, 192))
    return code.replace(needle, needle + f"        if (i + {distance} < nb) {{\n" + hints + "        }\n")


def f16c(code):
    return code.replace("GGML_CPU_FP16_TO_FP32(x[i].d)", "_cvtsh_ss(x[i].d)")


def compile_variants(out, variants):
    out.mkdir(exist_ok=False)
    entry = next(x for x in json.loads((BASE / "build/compile_commands.json").read_text())
                 if x["file"] == str(SOURCE))
    flags = shlex.split(entry["command"])
    flags = flags[:flags.index("-o")]
    manifest = {"baseline_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "base": str(BASE), "variants": [], "adopted": False}
    for name, code in variants:
        text = original[:avx] + code + original[end:]
        path = out / (name + ".c")
        path.write_text(text)
        (out / (name + ".patch")).write_text("".join(difflib.unified_diff(
            original.splitlines(True), text.splitlines(True),
            fromfile="a/ggml/src/ggml-cpu/arch/x86/quants.c",
            tofile="b/ggml/src/ggml-cpu/arch/x86/quants.c")))
        command = flags + ["-iquote", str(SOURCE.parent), "-shared", "-Wl,-Bsymbolic",
                           str(path), "-L" + str(BASE / "build/bin"),
                           "-Wl,-rpath," + str(BASE / "build/bin"),
                           "-lggml-cpu", "-lggml-base", "-lm", "-o", str(out / (name + ".so"))]
        with (out / (name + ".compile.log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        asm = subprocess.run(["objdump", "-d", "--disassemble=ggml_vec_dot_q6_K_q8_K",
                              str(out / (name + ".so"))], capture_output=True, text=True, check=True).stdout
        (out / (name + ".asm")).write_text(asm)
        manifest["variants"].append({"name": name, "command": command,
                                     "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                     "library_sha256": hashlib.sha256((out / (name + ".so")).read_bytes()).hexdigest(),
                                     "stack_operand_lines": sum("(%rsp)" in line for line in asm.splitlines())})
        (out / "prepare.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"compiled": name, "stack_operand_lines": manifest["variants"][-1]["stack_operand_lines"]}), flush=True)


if __name__ == "__main__":
    variants = [
        ("original", body),
        ("stream-scalar", streamed(False)),
        ("stream-vector", streamed(True)),
        ("original-f16c", f16c(body)),
        ("stream-vector-f16c", f16c(streamed(True))),
    ]
    variants += [(f"original-pf{d}", prefetch(body, d)) for d in (2, 4, 8)]
    variants += [(f"stream-vector-pf{d}", prefetch(streamed(True), d)) for d in (2, 4)]
    compile_variants(OUT, variants)

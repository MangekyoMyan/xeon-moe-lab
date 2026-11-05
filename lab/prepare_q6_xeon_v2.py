#!/usr/bin/env python3
"""Reuse widened scales and retain more independent dot-product work."""
import json
import re
import subprocess
from prepare_q6_xeon import LAB, BASE, body, compile_variants, prefetch

OUT = LAB / "q6-xeon-v2-20261020"


def scales16():
    code = body.replace("        int is = 0;\n", "")
    first = code.index("            const __m128i scale_0 =")
    last = code.index("\n            sumi_0", first)
    middle = ["            const __m128i scales16 = j == 0 ? scales_16_0 : scales_16_1;\n"]
    for v in range(8):
        mask = 0x0100 + v * 0x0202
        middle.append(f"            p16_{v} = _mm_madd_epi16(_mm_shuffle_epi8(scales16, _mm_set1_epi16(0x{mask:04x})), p16_{v});\n")
    return code[:first] + "".join(middle) + code[last:]


def pairs():
    code = body[:body.index("        int is = 0;")]
    code += "        for (int j = 0; j < 2; ++j) {\n"
    code += "            const __m128i scales16 = j == 0 ? scales_16_0 : scales_16_1;\n"
    for lane in (0, 1):
        code += f"            const __m128i hi_{lane} = _mm_loadu_si128((const __m128i *) (qh + j*32 + {lane*16}));\n"
    for group in range(4):
        code += "            {\n"
        for lane in (0, 1):
            off = (group % 2) * 32 + lane * 16
            code += f"                __m128i lo_{lane} = _mm_loadu_si128((const __m128i *) (q4 + j*64 + {off}));\n"
            if group >= 2:
                code += f"                lo_{lane} = _mm_srli_epi16(lo_{lane}, 4);\n"
            high = [
                f"_mm_slli_epi16(_mm_and_si128(hi_{lane}, m3), 4)",
                f"_mm_slli_epi16(_mm_and_si128(hi_{lane}, _mm_set1_epi8(12)), 2)",
                f"_mm_and_si128(hi_{lane}, _mm_set1_epi8(48))",
                f"_mm_srli_epi16(_mm_and_si128(hi_{lane}, _mm_set1_epi8(-64)), 2)",
            ][group]
            code += f"                const __m128i q_{lane} = _mm_or_si128(_mm_and_si128(lo_{lane}, m15), {high});\n"
            code += f"                const __m128i v_{lane} = _mm_loadu_si128((const __m128i *) (q8 + j*128 + {group*32+lane*16}));\n"
            mask = 0x0100 + (group * 2 + lane) * 0x0202
            code += f"                const __m128i scale_{lane} = _mm_shuffle_epi8(scales16, _mm_set1_epi16(0x{mask:04x}));\n"
            code += f"                sumi_{lane} = _mm_add_epi32(sumi_{lane}, _mm_madd_epi16(scale_{lane}, _mm_maddubs_epi16(q_{lane}, v_{lane})));\n"
        code += "            }\n"
    code += "        }\n"
    code += body[body.index("\n        sumi_0 = _mm_sub_epi32"):]
    return code


def main():
    variants = [("original", body), ("scales16", scales16()), ("pairs", pairs()),
                ("scales16-pf2", prefetch(scales16(), 2)), ("pairs-pf2", prefetch(pairs(), 2)),
                ("no-unroll", body.replace("        for (int j = 0;", "        #pragma GCC unroll 1\n        for (int j = 0;"))]
    compile_variants(OUT, variants)
    probe = (LAB / "q6-xeon-probe.c").read_text()
    probe = re.sub(r"static const char \* names\[\] = \{.*?\};",
                   "static const char * names[] = {" + ",".join(json.dumps(n) for n, _ in variants) + "};",
                   probe, flags=re.S)
    (OUT / "probe.c").write_text(probe)
    command = ["cc", "-std=gnu11", "-O3", "-march=native", "-fopenmp",
               "-I"+str(BASE/"source/ggml/include"), "-I"+str(BASE/"source/ggml/src"),
               str(OUT/"probe.c"), "-L"+str(BASE/"build/bin"), "-Wl,-rpath,"+str(BASE/"build/bin"),
               "-lggml-cpu", "-lggml-base", "-lm", "-ldl", "-o", str(OUT/"probe")]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()

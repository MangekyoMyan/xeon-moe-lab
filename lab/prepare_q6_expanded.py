#!/usr/bin/env python3
"""Test one-time, lossless Q6 unpacking before considering engine integration."""
import json
import re
import shlex
import subprocess
from prepare_q6_xeon import LAB, BASE, SOURCE, original, start, avx, end
from prepare_q6_xeon_v2 import scales16

OUT = LAB / "q6-expanded-20261020"
STRUCT = """
typedef struct {
    uint8_t qs[QK_K];
    int8_t scales[QK_K/16];
    float d;
} block_q6_expanded;
_Static_assert(sizeof(block_q6_expanded) == 276, "expanded block size");
"""
CONVERTER = """
static void expand(const block_q6_K * src, block_q6_expanded * dst, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        for (int j = 0; j < 2; ++j) {
            for (int g = 0; g < 4; ++g) {
                for (int t = 0; t < 32; ++t) {
                    uint8_t lo = src[b].ql[j*64+(g%2)*32+t];
                    if (g >= 2) lo >>= 4;
                    dst[b].qs[j*128+g*32+t] = (lo & 15) | (((src[b].qh[j*32+t] >> (g*2)) & 3) << 4);
                }
            }
        }
        memcpy(dst[b].scales, src[b].scales, sizeof(dst[b].scales));
        dst[b].d = ggml_fp16_to_fp32(src[b].d);
    }
}
"""


def main():
    OUT.mkdir(exist_ok=False)
    body = scales16()
    body = body.replace("GGML_CPU_FP16_TO_FP32(x[i].d)", "x[i].d")
    body = re.sub(r"    const __m128i m(?:3|15) =.*?\n", "", body)
    body = re.sub(r"        const uint8_t \* GGML_RESTRICT q(?:4|h) =.*?\n", "", body)
    a = body.index("            const __m128i q4bitsH_0")
    b = body.index("            const __m128i scales16 =", a)
    middle = []
    for v in range(8):
        middle.append(f"            __m128i p16_{v} = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i *) (x[i].qs + j*128 + {v*16})), _mm_loadu_si128((const __m128i *) (q8 + j*128 + {v*16})));\n")
    body = body[:a] + "".join(middle) + body[b:]
    header = original[start:avx].replace("const block_q6_K * GGML_RESTRICT x", "const block_q6_expanded * GGML_RESTRICT x")
    modified = original[:start] + STRUCT + header + body + original[end:]
    (OUT / "expanded.c").write_text(modified)
    entry = next(x for x in json.loads((BASE / "build/compile_commands.json").read_text())
                 if x["file"] == str(SOURCE))
    flags = shlex.split(entry["command"])
    flags = flags[:flags.index("-o")]
    command = flags + ["-iquote", str(SOURCE.parent), "-shared", "-Wl,-Bsymbolic",
                       str(OUT / "expanded.c"), "-L"+str(BASE/"build/bin"),
                       "-Wl,-rpath,"+str(BASE/"build/bin"), "-lggml-cpu", "-lggml-base",
                       "-lm", "-o", str(OUT/"expanded.so")]
    with (OUT/"compile.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # The unmodified kernel remains the comparator.
    (OUT/"original.so").symlink_to(LAB/"q6-xeon-20261020/original.so")
    probe = (LAB / "q6-xeon-probe.c").read_text()
    probe = re.sub(r"static const char \* names\[\] = \{.*?\};",
                   'static const char * names[] = {"original","expanded"};', probe, flags=re.S)
    needle = "static void exact(dot_fn * functions"
    probe = probe.replace(needle, STRUCT + CONVERTER + "\n" + needle)
    needle = "    float reference;\n"
    insertion = """    static char * guarded;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t capacity = (128*sizeof(block_q6_expanded)+page-1)/page*page;
    if (!guarded) {
        guarded = mmap(NULL, capacity+page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (guarded == MAP_FAILED || mprotect(guarded+capacity,page,PROT_NONE)) abort();
    }
    block_q6_expanded * expanded = (block_q6_expanded *) (guarded+capacity-(size_t)(n/QK_K)*sizeof(block_q6_expanded));
    expand(x,expanded,n/QK_K);
"""
    probe = probe.replace(needle, insertion + needle)
    probe = probe.replace("functions[v](n, &result, 0, x, 0, y, 0, 1);",
                          "functions[v](n, &result, 0, v ? (void *) expanded : x, 0, y, 0, 1);")
    needle = "    fill_y(y, k/QK_K, 777);\n"
    probe = probe.replace(needle, needle + """    size_t expanded_row = (size_t)(k/QK_K)*sizeof(block_q6_expanded);
    char * expanded = allocate(expanded_row*(size_t)m);
    expand((const block_q6_K *) x, (block_q6_expanded *) expanded, (size_t)(k/QK_K)*m);
""")
    probe = probe.replace("functions[v](k,result+i,0,x+i*row_bytes,0,y,0,1);",
                          "functions[v](k,result+i,0,v ? expanded+i*expanded_row : x+i*row_bytes,0,y,0,1);")
    probe = probe.replace("    free(x); free(y);", "    free(expanded); free(x); free(y);")
    (OUT/"probe.c").write_text(probe)
    probe_command = ["cc", "-std=gnu11", "-O3", "-march=native", "-fopenmp",
                     "-I"+str(BASE/"source/ggml/include"), "-I"+str(BASE/"source/ggml/src"),
                     str(OUT/"probe.c"), "-L"+str(BASE/"build/bin"), "-Wl,-rpath,"+str(BASE/"build/bin"),
                     "-lggml-cpu", "-lggml-base", "-lm", "-ldl", "-o", str(OUT/"probe")]
    subprocess.run(probe_command, check=True)
    (OUT/"prepare.json").write_text(json.dumps({"commands":[command,probe_command],
        "original_block_bytes":210,"expanded_block_bytes":276,
        "note":"Microbenchmark only; expansion outside timing, both representations retained; no engine integration."},indent=2)+"\n")
    print("Expanded Q6 probe compiled", flush=True)

if __name__ == "__main__":
    main()

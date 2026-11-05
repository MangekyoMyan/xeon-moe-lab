#!/usr/bin/env python3
"""Build isolated row-sharing kernels; the active runtime is not modified."""
import json
from pathlib import Path
import shlex
import subprocess

LAB = Path(__file__).resolve().parent
ROOT = LAB / "glm-q8-micro-20261020"
SOURCE = LAB / "llama-private-prefetch"
ROOT.mkdir(exist_ok=False)
path = SOURCE / "ggml/src/ggml-cpu/arch/x86/quants.c"
original = path.read_text()
start = original.index("void ggml_vec_dot_q8_0_q8_0_rows4(")
end = original.index("\n#endif", start)
fn = original[start:end]
variants = []
for rows, distance in ((2, 0), (2, 8), (4, 8)):
    name = f"glm_q8_rows{rows}_pf{distance}"
    variant = fn.replace("ggml_vec_dot_q8_0_q8_0_rows4", name)
    variant = variant.replace("[4]", f"[{rows}]").replace("r < 4", f"r < {rows}")
    if distance:
        anchor = "    for (; ib + 1 < nb; ib += 2) {\n"
        assert variant.count(anchor) == 1
        variant = variant.replace(anchor, anchor +
            f"        if (ib + {distance} < nb) {{\n"
            f"            for (int r = 0; r < {rows}; ++r) {{\n"
            f"                _mm_prefetch((const char *) &x[r][ib + {distance}], _MM_HINT_T0);\n"
            "            }\n"
            "        }\n")
    variants.append(variant)
text = original[:end] + "\n\n" + "\n\n".join(variants) + original[end:]
(ROOT / "quants.c").write_text(text)
entry = next(x for x in json.loads((LAB / "build-avx-prefetch/compile_commands.json").read_text())
             if x["file"] == str(path))
flags = shlex.split(entry["command"])
flags = flags[:flags.index("-o")]
lib = LAB / "build-avx-prefetch/bin"
command = flags + ["-iquote", str(path.parent), "-shared", "-Wl,-Bsymbolic",
                  str(ROOT / "quants.c"), "-L" + str(lib), "-Wl,-rpath," + str(lib),
                  "-lggml-cpu", "-lggml-base", "-lm", "-o", str(ROOT / "kernels.so")]
with (ROOT / "compile.log").open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
probe = ["cc", "-O3", "-march=native", "-fopenmp", "-std=gnu11",
         "-I" + str(SOURCE / "ggml/include"), "-I" + str(SOURCE / "ggml/src"),
         str(LAB / "glm-q8-row-probe.c"), "-L" + str(lib), "-Wl,-rpath," + str(lib),
         "-lggml-cpu", "-lggml-base", "-ldl", "-lm", "-o", str(ROOT / "probe")]
with (ROOT / "probe-compile.log").open("w") as log:
    subprocess.run(probe, stdout=log, stderr=subprocess.STDOUT, check=True)
(ROOT / "build.json").write_text(json.dumps({"commands": [command, probe]}, indent=2) + "\n")
print(ROOT)

"""Transplant supplied functions into an already reconstructed pristine fixed engine."""
import argparse, pathlib, shutil, subprocess

p=argparse.ArgumentParser()
p.add_argument("src",type=pathlib.Path)
p.add_argument("mode",choices=("base","mixed"))
a=p.parse_args()
src=a.src.resolve()
here=pathlib.Path(__file__).resolve().parent
owner=here.parent
xeon=src/"xeon-local"
# Apply only corrected kernel/header files. Keep the canonical ggml-cpu.c untouched until replacing the complete functions below.
cpu=src/"ggml/src/ggml-cpu/ggml-cpu.c"
original_cpu=cpu.read_text()
subprocess.run(["patch","--batch","--fuzz=0","-p1","-d",str(src),"-i",str(owner/"06-compact-signed-byte.patch")],check=True)
for file in ("mx-rhs.h","route37.h"): shutil.copy2(here/"excerpts"/file,xeon/file)
shutil.copy2(here/"fixture.cpp",xeon/"route37-test-fixture.cpp")
inc=xeon/"folded-input.inc"
with inc.open("a") as f:
    f.write("""
void xeon_iq_input_from_q8(const void * input, void * output, size_t count) {
    const auto * src = static_cast<const block_q8_K *>(input);
    auto * dst = static_cast<folded_block *>(output);
    for (size_t b=0;b<count;++b) fold_q8_block(src[b], dst[b]);
}
""")
with (xeon/"iq3-cache.h").open("a") as f:
    f.write("""
#ifdef __cplusplus
extern "C" {
#endif
const void * const * xeon_iq3_cache_lookup_experts_v2(const struct ggml_tensor * tensor);
void xeon_iq_input_from_q8(const void * input, void * output, size_t count);
#ifdef __cplusplus
}
#endif
""")
frag=src/"route37-test-extracts"
shutil.copytree(here/"excerpts",frag)
if a.mode == "mixed":
    subprocess.run(["patch","--batch","--fuzz=0","-p1","-d",str(frag),"-i",str(here/"08-mixed-dispatch.patch")],check=True)
s=original_cpu.replace('#include "iq3-cache.h"', '#include "iq3-cache.h"\n#include "mx-rhs.h"\n#include "route37.h"\n#define XEON_MX_PREPACK 0')
i=s.index("#define MMID_MATRIX_ROW")
j=s.index("/////////////////////////////////",i)
s=s[:i]+(frag/"mul_mat_id_route37.c").read_text()+"\n"+s[j:]
i=s.index("struct ggml_cplan ggml_graph_plan(")
j=s.index("    return cplan;\n}",i)+len("    return cplan;\n}")
s=s[:i]+(frag/"graph_plan_route37.c").read_text()+s[j:]
cpu.write_text(s)
with (src/"CMakeLists.txt").open("a") as f:
    f.write('\ntarget_sources(ggml-cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/route37-test-fixture.cpp")\n')

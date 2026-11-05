from pathlib import Path
import shutil, hashlib, json
LAB=Path('/home/codex/projects/llm-lab')
ROOT=LAB/'review-full-replica-20261020'; OUT=LAB/'results/full-replica-20261020'
ROOT.mkdir(); OUT.mkdir()
src=ROOT/'source'
shutil.copytree(LAB/'llama-private-prefetch',src,ignore=shutil.ignore_patterns('.git'))
files=['ggml/src/ggml-cpu/numa-replica.cpp','ggml/src/ggml-cpu/numa-replica.h','ggml/src/ggml-cpu/ggml-cpu.c']
(OUT/'original-hashes.json').write_text(json.dumps({n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in files},indent=2))
p=src/files[0];s=p.read_text()
s=s.replace('namespace {\nstruct replica_context;', '''namespace {
static int full_mode() {
    static const int mode = [] {
        const char * env = std::getenv("GGML_CPU_NUMA_FULL");
        return env && std::strcmp(env, "1") == 0 ? 1 : env && std::strcmp(env, "2") == 0 ? 2 : 0;
    }();
    return mode;
}
struct replica_context;''')
s=s.replace('    bool trace;','    bool trace;\n    int copies;')
s=s.replace('explicit replica_context(size_t size) : size(size), traits(this)', 'explicit replica_context(size_t size) : size(size), copies(full_mode() == 2 ? 1 : 2), traits(this)')
s=s.replace('tensor->extra = &context->traits;', 'tensor->extra = full_mode() ? nullptr : &context->traits;')
s=s.replace('node < 2; ++node)', 'node < context->copies; ++node)')
s=s.replace('// Bound this private experiment to the profiled 3 GiB weight set.\n    if (size > size_t(4) * 1024 * 1024 * 1024)', '// Bound this private experiment to one small model per buffer.\n    if (size > size_t(full_mode() ? 32 : 4) * 1024 * 1024 * 1024)')
s=s.replace('const unsigned long mask = 1UL << node;\n        if (syscall(SYS_mbind, data, size, MPOL_BIND,', 'const unsigned long mask = full_mode() == 2 ? 3UL : 1UL << node;\n        if (syscall(SYS_mbind, data, size, full_mode() == 2 ? MPOL_INTERLEAVE : MPOL_BIND,')
s=s.replace('bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {', '''bool supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) override {
        if (full_mode()) {
            ggml_tensor plain = *op;
            ggml_tensor sources[GGML_MAX_SRC];
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (op->src[i]) {
                    sources[i] = *op->src[i];
                    sources[i].buffer = nullptr;
                    sources[i].extra = nullptr;
                    plain.src[i] = &sources[i];
                }
            }
            return ggml_backend_dev_supports_op(dev, &plain);
        }''')
s=s.replace('ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {', 'ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {\n        if (full_mode()) return nullptr;')
s=s.replace('if (!enabled) return nullptr;', 'if (!enabled && !full_mode()) return nullptr;')
pos=s.index('\n#else\nggml_backend_buffer_type_t')
s=s[:pos]+'''

bool ggml_cpu_numa_replica_localize(const ggml_tensor * op, ggml_tensor * local, ggml_tensor * sources) {
    if (!full_mode()) return false;
    bool changed = false;
    unsigned node = 2;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = op->src[i];
        if (!src) continue;
        auto * buffer = src->buffer;
        if (!buffer && src->view_src) buffer = src->view_src->buffer;
        if (!buffer || buffer->buft != ggml_backend_cpu_numa_replica_buffer_type()) continue;
        auto * context = static_cast<replica_context *>(buffer->context);
        if (!changed) {
            *local = *op;
            if (syscall(SYS_getcpu, nullptr, &node, nullptr) != 0 || node > 1) node = 2;
            changed = true;
        }
        const size_t offset = tensor_offset(context, src, 0, ggml_nbytes(src));
        sources[i] = *src;
        sources[i].data = static_cast<char *>(context->data[context->copies == 2 && node < 2 ? node : 0]) + offset;
        local->src[i] = &sources[i];
        if (context->trace) context->calls[node].fetch_add(1, std::memory_order_relaxed);
    }
    return changed;
}
''' + s[pos:]
s=s.replace('#endif\n', 'bool ggml_cpu_numa_replica_localize(const ggml_tensor *, ggml_tensor *, ggml_tensor *) { return false; }\n#endif\n')
p.write_text(s)
p=src/files[1];p.write_text('''#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
ggml_backend_buffer_type_t ggml_backend_cpu_numa_replica_buffer_type(void);
bool ggml_cpu_numa_replica_localize(const struct ggml_tensor * op, struct ggml_tensor * local, struct ggml_tensor * sources);
#ifdef __cplusplus
}
#endif
''')
p=src/files[2];s=p.read_text();s='#include "numa-replica.h"\n'+s
needle='    // extra_buffer op?'
assert s.count(needle)==1
s=s.replace(needle, '''    struct ggml_tensor local;
    struct ggml_tensor sources[GGML_MAX_SRC];
    if (ggml_cpu_numa_replica_localize(tensor, &local, sources)) {
        tensor = &local;
    }

'''+needle)
needle='                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);'
s=s.replace(needle,'''                struct ggml_tensor local;
                struct ggml_tensor sources[GGML_MAX_SRC];
                if (ggml_cpu_numa_replica_localize(mul_node, &local, sources)) {
                    mul_node = &local;
                }
'''+needle)
p.write_text(s)
print(ROOT)

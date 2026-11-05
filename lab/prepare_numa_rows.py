#!/usr/bin/env python3
"""Prepare an isolated row-local NUMA candidate from the active private source."""
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil

lab = Path(__file__).resolve().parent
base = lab / 'llama-private-prefetch'
root = lab / 'review-numa-rows-20261020'
out = lab / 'results/optimization-numa-rows-20261020'
root.mkdir(exist_ok=False)
out.mkdir(exist_ok=False)
shutil.copytree(base, root / 'source')
files = {}

def edit(relative, transform):
    before = (base / relative).read_text()
    after = transform(before)
    assert before != after
    (root / 'source' / relative).write_text(after)
    files[relative] = {'before': hashlib.sha256(before.encode()).hexdigest(),
                      'after': hashlib.sha256(after.encode()).hexdigest()}
    with (root / 'candidate.patch').open('a') as stream:
        stream.write(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/' + relative, tofile='b/' + relative)))

def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

def cpu(text):
    text = once(text, '''void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {''', '''static void ggml_compute_forward_mul_mat_impl(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst,
              int64_t row_start,
              int64_t row_end) {''')
    text = once(text, '''    if (ggml_compute_forward_mul_mat_tiled(params, dst)) {''',
        '''    if (row_start < 0 && ggml_compute_forward_mul_mat_tiled(params, dst)) {''')
    anchor = '''    // Now select a reasonable chunk size.
    int chunk_size = 16;'''
    text = once(text, anchor, '''    if (row_start >= 0) {
        GGML_ASSERT(nr1 == 1 && row_start <= row_end && row_end <= nr0);
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, 1, row_start, row_end, 0, 1);
        return;
    }

''' + anchor)
    anchor = '// ggml_compute_forward_mul_mat_id\n'
    text = once(text, anchor, '''void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    ggml_compute_forward_mul_mat_impl(params, dst, -1, -1);
}

void ggml_compute_forward_mul_mat_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q8_0 && ggml_n_dims(dst->src[0]) == 2);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32 && ggml_nrows(dst->src[1]) == 1);
    const int64_t tiles = (dst->ne[0] + 15) / 16;
    const int64_t start = 16 * (tiles * params->ith / params->nth);
    const int64_t end = MIN(16 * (tiles * (params->ith + 1) / params->nth), dst->ne[0]);
    ggml_compute_forward_mul_mat_impl(params, dst, start, end);
}

''' + anchor)
    return text

def header(text):
    anchor = 'void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);'
    return once(text, anchor, anchor + '\nvoid ggml_compute_forward_mul_mat_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);')

def numa(text):
    text = once(text, 'namespace {\nstruct replica_context;', '''namespace {
static bool shard_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CPU_NUMA_SHARD");
        return env && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

struct replica_context;''')
    text = once(text, '    bool trace;\n', '    bool trace;\n    bool shard = shard_enabled();\n')
    text = once(text, '''    unsigned node = 2;''', '''    if (context->shard) {
        if (op->src[1]->type == GGML_TYPE_F32 && ggml_nrows(op->src[1]) == 1 &&
                ggml_n_dims(op->src[0]) == 2 && !params->use_ref) {
            if (context->trace) {
                unsigned node = 2;
                if (syscall(SYS_getcpu, nullptr, &node, nullptr) != 0 || node > 1) node = 2;
                context->calls[node].fetch_add(1, std::memory_order_relaxed);
            }
            ggml_compute_forward_mul_mat_rows(params, op);
        } else {
            ggml_compute_forward_mul_mat(params, op);
        }
        return true;
    }
    unsigned node = 2;''')
    text = once(text, '''    tensor->extra = &context->traits;
    if (context->trace) {''', '''    tensor->extra = &context->traits;
    if (context->shard && eligible(tensor)) {
        const size_t bytes = ggml_nbytes(tensor);
        const size_t page_size = size_t(sysconf(_SC_PAGESIZE));
        const size_t start = tensor_offset(context, tensor, 0, bytes);
        if (page_size != 4096 || start % page_size || bytes % (2 * page_size)) {
            GGML_LOG_ERROR("NUMA_SHARD requires page-aligned weight halves\\n");
            return GGML_STATUS_FAILED;
        }
        for (unsigned node = 0; node < 2; ++node) {
            void * ptr = static_cast<char *>(tensor->data) + node * (bytes / 2);
            const unsigned long mask = 1UL << node;
            if (syscall(SYS_mbind, ptr, bytes / 2, MPOL_BIND, &mask, sizeof(mask) * 8, 0) != 0) {
                GGML_LOG_ERROR("NUMA_SHARD mbind failed: %s\\n", std::strerror(errno));
                return GGML_STATUS_FAILED;
            }
            if (context->trace) {
                std::fprintf(stderr, "NUMA_SHARD_RANGE node=%u address=%p bytes=%zu name=%s\\n",
                    node, ptr, bytes / 2, tensor->name);
            }
        }
    }
    if (context->trace) {'''.replace('\\n"', '\\\\n"'))
    # The release path still visits both pointers, with the unused pointer null.
    text = text.replace('for (int node = 0; node < 2; ++node) {\n        std::memcpy',
                        'for (int node = 0; node < (context->shard ? 1 : 2); ++node) {\n        std::memcpy')
    text = text.replace('for (int node = 0; node < 2; ++node) std::memset',
                        'for (int node = 0; node < (context->shard ? 1 : 2); ++node) std::memset')
    text = text.replace('for (int node = 0; node < 2; ++node) {\n        if (context->size)',
                        'for (int node = 0; node < (context->shard ? 1 : 2); ++node) {\n        if (context->size)')
    text = once(text, 'for (int node = 0; size && node < 2; ++node) {',
                      'for (int node = 0; size && node < (context->shard ? 1 : 2); ++node) {')
    text = once(text, '''        if (syscall(SYS_mbind, data, size, MPOL_BIND, &mask, sizeof(mask) * 8, 0) != 0) {''',
        '''        if (!context->shard && syscall(SYS_mbind, data, size, MPOL_BIND, &mask, sizeof(mask) * 8, 0) != 0) {''')
    text = once(text, 'static const char * buffer_name(ggml_backend_buffer_type_t) { return "CPU_NUMA_REPLICA"; }',
        'static const char * buffer_name(ggml_backend_buffer_type_t) { return shard_enabled() ? "CPU_NUMA_SHARD" : "CPU_NUMA_REPLICA"; }')
    text = once(text, 'static size_t buffer_alignment(ggml_backend_buffer_type_t) { return TENSOR_ALIGNMENT; }',
        'static size_t buffer_alignment(ggml_backend_buffer_type_t) { return shard_enabled() ? 4096 : TENSOR_ALIGNMENT; }')
    text = once(text, '    if (!enabled) return nullptr;', '    if (!enabled && !shard_enabled()) return nullptr;')
    return text

edit('ggml/src/ggml-cpu/ggml-cpu.c', cpu)
edit('ggml/src/ggml-cpu/ops.h', header)
edit('ggml/src/ggml-cpu/numa-replica.cpp', numa)
active = [lab / 'run_deepseek_flash.sh', lab / 'rebuild_deepseek.sh',
          lab / 'build-avx-prefetch/bin/llama-bench',
          lab / 'build-avx-prefetch/bin/llama-completion',
          lab / 'build-avx-prefetch/bin/libggml-cpu.so',
          lab / 'experiment-preferences.json']
record = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'baseline_source': str(base), 'candidate': str(root), 'files': files,
          'active_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in active},
          'adopted': False,
          'scope': 'Selected 107 Q8 weights: one copy, first/second row halves on nodes 0/1. Static 16-row tiles in single-column multiply. Existing arithmetic and graph barriers retained. All other weights and MoE scheduling unchanged.',
          'placement_contract': 'Performance requires the existing strict 12-worker CPUs 0-11 mapping; correctness does not depend on physical worker placement. Multi-column operations retain the general scheduler.',
          'planned_controls': ['current default', 'old full replica', 'shard plus fixed rows', 'fixed rows with interleaved copied weights']}
(out / 'experiment.json').write_text(json.dumps(record, indent=2) + '\n')
(root / 'prepare.json').write_text(json.dumps(record, indent=2) + '\n')
driver = (lab / 'run_cpu_ab_light.py').read_text()
driver = driver.replace("'GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_PAIR'", "'GGML_CPU_NUMA_REPLICA', 'GGML_CPU_NUMA_SHARD', 'GGML_CPU_MOE_PAIR'")
(lab / 'run_cpu_ab_numa_rows.py').write_text(driver)
probe = (lab / 'numa-replica-probe.cpp').read_text()
probe = probe.replace('"CPU_NUMA_REPLICA"', '"CPU_NUMA_SHARD"').replace('bytes + 256', 'bytes + 8192').replace('+ 128)', '+ 4096)')
probe = probe.replace('    ggml_cpu_init();', '    ggml_cpu_init();')
(out / 'numa-shard-probe.cpp').write_text(probe)
print(json.dumps({'prepared': str(root), 'output': str(out)}))

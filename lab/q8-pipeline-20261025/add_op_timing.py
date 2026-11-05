#!/usr/bin/env python3
"""Optional bounded CPU-op timing; diagnostic runs are excluded from speed A/B."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CPU = ROOT.parent / 'review-q8-reuse-20261025/source/ggml/src/ggml-cpu/ggml-cpu.c'

CODE = r'''
// Private diagnostic mode. Only worker zero records intervals; the table is
// bounded and mutex-protected for callers that use more than one threadpool.
// Timing includes internal operation barriers and the following graph barrier.
static bool ggml_cpu_op_timing = false;
#if defined(__linux__)
struct ggml_cpu_op_record {
    char key[192];
    uint64_t calls;
    uint64_t total_us;
    uint64_t compute_us;
    uint64_t min_us;
    uint64_t max_us;
};
static struct ggml_cpu_op_record ggml_cpu_op_records[1024];
static pthread_mutex_t ggml_cpu_op_record_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t ggml_cpu_op_dropped = 0;

static void ggml_cpu_op_timing_dump(void) {
    pthread_mutex_lock(&ggml_cpu_op_record_mutex);
    fprintf(stderr, "OP_TIMING_HEADER\toperation\tweight\ttype\tK\tM\tN1\tN2\texperts\tfused\tcalls\ttotal_us\tcompute_us\tmin_us\tmax_us\n");
    for (size_t i = 0; i < 1024; ++i) {
        const struct ggml_cpu_op_record * r = ggml_cpu_op_records + i;
        if (r->calls) fprintf(stderr, "OP_TIMING\t%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
            r->key, r->calls, r->total_us, r->compute_us, r->min_us, r->max_us);
    }
    fprintf(stderr, "OP_TIMING_DROPPED\t%" PRIu64 "\n", ggml_cpu_op_dropped);
    pthread_mutex_unlock(&ggml_cpu_op_record_mutex);
}

static void ggml_cpu_op_timing_add(const struct ggml_tensor * node, int fused, int64_t started, int64_t computed) {
    const int64_t ended = ggml_time_us();
    char key[192];
    const bool mat = node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID;
    if (mat) {
        const struct ggml_tensor * weight = node->src[0];
        const char * name = weight->name;
        if (strncmp(name, "blk.", 4) == 0) {
            const char * suffix = strchr(name + 4, '.');
            if (suffix) name = suffix + 1;
        }
        snprintf(key, sizeof(key), "%s\t%.64s\t%s\t%" PRId64 "\t%" PRId64 "\t%" PRId64 "\t%" PRId64 "\t%" PRId64 "\t%d",
            ggml_op_name(node->op), name, ggml_type_name(weight->type), weight->ne[0], weight->ne[1],
            node->ne[1], node->ne[2], weight->ne[2], fused);
    } else {
        snprintf(key, sizeof(key), "%s\t-\t-\t0\t0\t0\t0\t0\t%d", ggml_op_name(node->op), fused);
    }
    uint32_t hash = 2166136261u;
    for (const unsigned char * p = (const unsigned char *) key; *p; ++p) hash = (hash ^ *p) * 16777619u;
    pthread_mutex_lock(&ggml_cpu_op_record_mutex);
    for (size_t probe = 0; probe < 1024; ++probe) {
        struct ggml_cpu_op_record * r = ggml_cpu_op_records + ((hash + probe) % 1024);
        if (!r->calls || strcmp(r->key, key) == 0) {
            const uint64_t total = (uint64_t) MAX(ended - started, 0);
            if (!r->calls) {
                memcpy(r->key, key, strlen(key) + 1);
                r->min_us = total;
            }
            ++r->calls;
            r->total_us += total;
            r->compute_us += (uint64_t) MAX(computed - started, 0);
            r->min_us = MIN(r->min_us, total);
            r->max_us = MAX(r->max_us, total);
            pthread_mutex_unlock(&ggml_cpu_op_record_mutex);
            return;
        }
    }
    ++ggml_cpu_op_dropped;
    pthread_mutex_unlock(&ggml_cpu_op_record_mutex);
}
#else
static void ggml_cpu_op_timing_add(const struct ggml_tensor * node, int fused, int64_t started, int64_t computed) {
    GGML_UNUSED(node); GGML_UNUSED(fused); GGML_UNUSED(started); GGML_UNUSED(computed);
}
#endif

'''

def main():
    text = CPU.read_text()
    assert 'OP_TIMING_HEADER' not in text
    anchor = 'static int ggml_cpu_q8_moe_schedule = 0;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, CODE + anchor)
    anchor = '        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '        const int64_t op_started = ggml_cpu_op_timing && state->ith == 0 ? ggml_time_us() : 0;\n' + anchor)
    anchor = '        if (state->ith == 0 && cplan->abort_callback &&'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '        const int64_t op_computed = op_started ? ggml_time_us() : 0;\n\n' + anchor)
    anchor = '''        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''        if (node_n + 1 < cgraph->n_nodes || ggml_cpu_op_timing) {
            ggml_barrier(state->threadpool);
        }
        if (op_started) ggml_cpu_op_timing_add(node, n_fused, op_started, op_computed);''')
    anchor = '            const char * cols2 = getenv("GGML_CPU_Q8_COLS2");'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''#if defined(__linux__)
            const char * op_timing = getenv("GGML_CPU_OP_TIMING");
            ggml_cpu_op_timing = op_timing && strcmp(op_timing, "1") == 0;
            if (ggml_cpu_op_timing) atexit(ggml_cpu_op_timing_dump);
#endif
''' + anchor)
    CPU.write_text(text)
    print('Added optional CPU operation timing; rebuild required')

if __name__ == '__main__':
    main()

"""Prepare an isolated mixed OpenMP/pthread executor; no builds or runtime changes."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;OUT=R/'hybrid-pool'
def main():
    OUT.mkdir(exist_ok=True)
    source=R/'mx-prepack/backend/ggml-cpu.c';s=source.read_text()
    def once(old,new):
        nonlocal s
        assert s.count(old)==1,(old[:80],s.count(old));s=s.replace(old,new,1)
    def section(start,end,fn):
        nonlocal s
        a=s.index(start);b=s.index(end,a);part=s[a:b];s=s[:a]+fn(part)+s[b:]
    once('struct ggml_threadpool {\n','struct ggml_threadpool {\n    bool use_openmp; // Immutable private mode; public ABI unchanged.\n')
    once('#ifndef GGML_USE_OPENMP\n    ggml_thread_t thrd;\n    int  last_graph;\n    bool pending;\n#endif',
         '    ggml_thread_t thrd;\n    int  last_graph;\n    bool pending;')
    section('void ggml_barrier(', 'void ggml_threadpool_chunk_set(',lambda p:p.replace('#ifdef GGML_USE_OPENMP\n    #pragma omp barrier\n#else','    if (tp->use_openmp) {\n        #pragma omp barrier\n        return;\n    }',1).replace('\n#endif\n}', '\n}',1))
    section('void ggml_threadpool_free(', 'struct ggml_cplan ggml_graph_plan(',lambda p:p
        .replace('#ifndef GGML_USE_OPENMP\n    struct ggml_compute_state* workers', '    if (!threadpool->use_openmp) {\n    struct ggml_compute_state* workers',1)
        .replace('#endif // GGML_USE_OPENMP', '    }',1)
        .replace('#ifndef GGML_USE_OPENMP\n// pause/resume','// pause/resume',1)
        .replace('}\n#endif\n\nvoid ggml_threadpool_pause', '}\n\nvoid ggml_threadpool_pause',1)
        .replace('#ifndef GGML_USE_OPENMP\n    ggml_mutex_lock', '    if (!threadpool || threadpool->use_openmp) return;\n    ggml_mutex_lock')
        .replace('#else\n    UNUSED(threadpool);\n#endif',''))
    for action in ['start','done']:
        old=f'''#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-{action} cplan %p\\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-{action} cplan %p last-graph %d\\n", state->ith, (const void *)cplan, state->last_graph);
#endif'''
        once(old,f'    GGML_PRINT_DEBUG("thread #%d compute-{action} cplan %p last-graph %d\\n", state->ith, (const void *)cplan, state->last_graph);')
    once('#ifndef GGML_USE_OPENMP\n\n// check if thread is ready','// check if thread is ready')
    once('#endif // GGML_USE_OPENMP\n\nstatic struct ggml_threadpool * ggml_threadpool_new_impl', 'static struct ggml_threadpool * ggml_threadpool_new_impl')
    old='''    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);'''
    once(old,'''    const int n_graph = xeon_next_packed_graph(
        atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed), n_threads);''')
    once('#include "mx-rhs.h"','#include "mx-rhs.h"\n#include "packed-generation.h"')
    old='''       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

'''
    once(old,'')
    once('''    // Always take the mutex here because the worker threads are doing hybrid poll/wait
''','''    // The caller may have just executed an OpenMP graph on another CPU.
    // Rebind even on the first native graph and active-one-thread graphs.
    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
        ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
    }

    // Always take the mutex here because the worker threads are doing hybrid poll/wait
''')
    once('''    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));''','''    const char * policy = getenv("GGML_CPU_HYBRID_POOL");
    GGML_ASSERT(!policy || strcmp(policy, "0") == 0 || strcmp(policy, "1") == 0);
    const bool native = policy && strcmp(policy, "1") == 0 && cgraph == NULL && cplan == NULL && tpp->n_threads == 24;
    GGML_ASSERT(tpp->n_threads > 0 && tpp->n_threads <= GGML_MAX_N_THREADS);
    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));''')
    once('''        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;''','''        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->use_openmp       = !native;
        threadpool->cgraph           = cgraph;''')
    section('static struct ggml_threadpool * ggml_threadpool_new_impl(', '\nstruct ggml_threadpool * ggml_threadpool_new(',lambda p:p
        .replace('#ifdef GGML_USE_OPENMP','    if (threadpool->use_openmp) {',1)
        .replace('#else // GGML_USE_OPENMP','    } else {',1)
        .replace('#endif // GGML_USE_OPENMP','    }',1))
    section('enum ggml_status ggml_graph_compute(', '\nenum ggml_status ggml_graph_compute_with_ctx(',lambda p:p
        .replace('#ifdef GGML_USE_OPENMP','    if (threadpool->use_openmp) {',1)
        .replace('#else\n    if (n_threads >', '    } else {\n    if (n_threads >',1)
        .replace('\n#endif\n\n    // don\'t leave affinity','\n    }\n\n    // don\'t leave affinity',1)
        .replace('''        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread''','''        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
        ggml_graph_compute_thread''',1))
    capacity='''    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }
'''
    assert s.count(capacity)==1
    s=s.replace(capacity,'',1)
    section('enum ggml_status ggml_graph_compute(', '\nenum ggml_status ggml_graph_compute_with_ctx(',lambda p:p.replace('    if (threadpool->use_openmp) {',capacity+'\n    if (threadpool->use_openmp) {',1))
    once('''struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {''','''GGML_BACKEND_API int xeon_threadpool_native_mode_v1(const struct ggml_threadpool * threadpool);
int xeon_threadpool_native_mode_v1(const struct ggml_threadpool * threadpool) {
    return threadpool && !threadpool->use_openmp;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {''')
    # Only compile-time feature include/initialization branches should remain.
    assert s.count('#ifdef GGML_USE_OPENMP')==2
    assert '#ifndef GGML_USE_OPENMP' not in s
    (OUT/'ggml-cpu.c').write_text(s)
    header='''#pragma once
#include <stdint.h>
#include <string.h>
// Preserve the existing private signed atomic transport without signed shifts.
static inline int xeon_next_packed_graph(int previous, int threads) {
#ifdef __cplusplus
    static_assert(sizeof(int) == sizeof(uint32_t));
#else
    _Static_assert(sizeof(int) == sizeof(uint32_t), "32-bit private transport required");
#endif
    uint32_t old = (uint32_t) previous;
    uint32_t packed = ((old + UINT32_C(0x10000)) & UINT32_C(0xffff0000)) |
                      ((uint32_t) threads & UINT32_C(0xffff));
    int result;
    memcpy(&result, &packed, sizeof(result));
    return result;
}
'''
    (OUT/'packed-generation.h').write_text(header)
    (OUT/'PREPARATION.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256((OUT/'ggml-cpu.c').read_bytes()).hexdigest(),'scope':'private immutable pool mode; native persistent24 only when explicitly enabled; disposable/other pools OpenMP; unsigned generation; explicit master affinity'},indent=2)+'\n')
    print('Prepared isolated hybrid executor source; not compiled.',flush=True)
if __name__=='__main__':main()

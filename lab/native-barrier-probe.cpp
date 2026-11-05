#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <unistd.h>

struct alignas(64) slot {
    uint64_t value;
};

struct shared_state {
    ggml_threadpool * pool;
    slot slots[24];
    std::atomic<int> errors{0};
    int run = 0;
};

static void exchange(ggml_tensor * dst, const ggml_tensor *, int ith, int nth, void * opaque) {
    auto & s = *static_cast<shared_state *>(opaque);
    for (int iteration = 0; iteration < 4000; ++iteration) {
        const uint64_t base = uint64_t(s.run) * 1000000 + uint64_t(iteration) * 131;
        s.slots[ith].value = base + ith;
        if (ith == 0 && iteration % 257 == 0) usleep(1);
        ggml_barrier(s.pool);
        for (int i = 0; i < nth; ++i) {
            if (s.slots[i].value != base + i) s.errors.fetch_add(1, std::memory_order_relaxed);
        }
        ggml_barrier(s.pool);
    }
    for (int i = ith; i < 24; i += nth) static_cast<float *>(dst->data)[i] = float(s.run * 24 + i);
}

static bool abort_after_node(void * opaque) {
    return ++*static_cast<int *>(opaque) == 1;
}

int main() {
    ggml_cpu_init();
    auto tpp = ggml_threadpool_params_default(24);
    tpp.strict_cpu = true;
    for (int i = 0; i < 24; ++i) tpp.cpumask[i] = true;
    shared_state state{};
    state.pool = ggml_threadpool_new(&tpp);
    if (!state.pool) return 1;
    auto * ctx = ggml_init({4u * 1024u * 1024u, nullptr, false});
    if (!ctx) return 1;
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
    ggml_set_f32(input, 0.0f);
    auto * a = ggml_map_custom1(ctx, input, exchange, GGML_N_TASKS_MAX, &state);
    auto * b = ggml_map_custom1(ctx, a, exchange, GGML_N_TASKS_MAX, &state);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, b);
    for (int threads : {12, 24, 6, 2, 1, 12, 24, 12}) {
        for (bool abort : {false, true}) {
            ++state.run;
            auto plan = ggml_graph_plan(graph, threads, state.pool);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            int calls = 0;
            if (abort) {
                plan.abort_callback = abort_after_node;
                plan.abort_callback_data = &calls;
            }
            const auto status = ggml_graph_compute(graph, &plan);
            if (status != (abort ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS) || state.errors.load()) return 2;
            auto * output = abort ? a : b;
            for (int i = 0; i < 24; ++i) {
                if (static_cast<float *>(output->data)[i] != float(state.run * 24 + i)) return 3;
            }
            std::printf("PASS threads=%d abort=%d run=%d\n", threads, abort, state.run);
        }
    }
    ggml_free(ctx);
    ggml_threadpool_free(state.pool);
    std::puts("PASS 16 cases, 192000 barrier rounds including single-thread cases, no visibility errors");
    return 0;
}

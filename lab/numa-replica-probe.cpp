#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static unsigned rng = 0x371031;
static unsigned next() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    ggml_cpu_init();
    auto * reg = ggml_backend_cpu_reg();
    auto * device = ggml_backend_reg_dev_get(reg, 0);
    using get_buffers_t = ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t);
    auto get_buffers = reinterpret_cast<get_buffers_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    if (!get_buffers) return 2;
    ggml_backend_buffer_type_t buft = nullptr;
    for (auto ** p = get_buffers(device); p && *p; ++p)
        if (std::strcmp(ggml_backend_buft_name(*p), "CPU_NUMA_REPLICA") == 0) buft = *p;
    if (!buft) return 3;
    std::ofstream output(argv[1], std::ios::binary);
    int cases = 0;
    const int shapes[][2] = {{1024, 8192}, {1024, 32768}, {8192, 4096}};
    for (const auto & s : shapes) {
        auto * meta = ggml_init({1024 * 1024, nullptr, true});
        auto * ref = ggml_new_tensor_2d(meta, GGML_TYPE_Q8_0, s[0], s[1]);
        auto * replica = ggml_dup_tensor(meta, ref);
        ggml_set_name(replica, "numa_replica_probe");
        const size_t bytes = ggml_nbytes(ref);
        auto * plain_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes + 256);
        auto * replica_buffer = ggml_backend_buft_alloc_buffer(buft, bytes + 256);
        if (!plain_buffer || !replica_buffer) return 4;
        ggml_backend_buffer_clear(replica_buffer, 0x37);
        if (ggml_backend_tensor_alloc(plain_buffer, ref, static_cast<char *>(ggml_backend_buffer_get_base(plain_buffer)) + 128) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(replica_buffer, replica, static_cast<char *>(ggml_backend_buffer_get_base(replica_buffer)) + 128) != GGML_STATUS_SUCCESS) return 5;
        std::vector<unsigned char> values(bytes), restored(bytes);
        const size_t block_bytes = ggml_type_size(GGML_TYPE_Q8_0);
        for (size_t offset = 0; offset < bytes; offset += block_bytes) {
            const ggml_fp16_t scale = ggml_fp32_to_fp16(float(next() % 7 + 1) / 64.0f);
            std::memcpy(values.data() + offset, &scale, sizeof(scale));
            for (size_t j = 2; j < block_bytes; ++j) values[offset + j] = next();
        }
        ggml_backend_tensor_set(ref, values.data(), 0, bytes);
        ggml_backend_tensor_copy(ref, replica);
        // Exercise partial writes and their propagation to both NUMA copies.
        for (size_t j = 3; j < 13; ++j) values[j] = 0x21;
        ggml_backend_tensor_memset(replica, 0x21, 3, 10);
        for (size_t j = block_bytes + 5; j < block_bytes + 13; ++j) values[j] = 0x17;
        ggml_backend_tensor_set(replica, values.data() + block_bytes + 5, block_bytes + 5, 8);
        ggml_backend_tensor_get(replica, restored.data(), 0, bytes);
        if (restored != values) return 6;
        ggml_backend_tensor_set(ref, values.data(), 0, bytes);
        for (int n : {1, 8, 32}) {
            auto * ctx = ggml_init({32u * 1024u * 1024u, nullptr, false});
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s[0], n);
            for (int64_t i = 0; i < ggml_nelements(x); ++i)
                static_cast<float *>(x->data)[i] = float(int(next() % 257) - 128) / 32.0f;
            auto * a = ggml_mul_mat(ctx, ref, x);
            auto * b = ggml_mul_mat(ctx, replica, x);
            if (!ggml_backend_dev_supports_op(device, b)) return 7;
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, a);
            ggml_build_forward_expand(graph, b);
            const int placements[][2] = {{1, 0}, {1, 6}, {6, 0}, {6, 6}, {12, 0}, {24, 0}};
            for (const auto & placement : placements) {
                auto params = ggml_threadpool_params_default(placement[0]);
                params.strict_cpu = true;
                for (int i = placement[1]; i < placement[1] + placement[0]; ++i) params.cpumask[i] = true;
                auto * pool = ggml_threadpool_new(&params);
                auto plan = ggml_graph_plan(graph, placement[0], pool);
                std::vector<uint8_t> scratch(plan.work_size);
                plan.work_data = scratch.data();
                if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) return 8;
                if (std::memcmp(a->data, b->data, ggml_nbytes(a))) {
                    std::fprintf(stderr, "Mismatch K=%d M=%d N=%d threads=%d first_cpu=%d\n", s[0], s[1], n, placement[0], placement[1]);
                    return 9;
                }
                for (int64_t i = 0; i < ggml_nelements(a); ++i)
                    if (!std::isfinite(static_cast<float *>(a->data)[i])) return 10;
                output.write(static_cast<char *>(b->data), ggml_nbytes(b));
                if (!output) return 11;
                ggml_threadpool_free(pool);
                ++cases;
            }
            ggml_free(ctx);
        }
        ggml_backend_buffer_free(replica_buffer);
        ggml_backend_buffer_free(plain_buffer);
        ggml_free(meta);
    }
    std::printf("PASS %d graph comparisons with partial buffer writes, node0/node1/dual/SMT placement\n", cases);
    return 0;
}

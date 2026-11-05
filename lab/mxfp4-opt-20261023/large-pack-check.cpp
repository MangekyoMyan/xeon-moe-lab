#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <omp.h>

int main() {
    assert(std::getenv("GGML_CPU_MXFP4_PACKED") && !std::strcmp(std::getenv("GGML_CPU_MXFP4_PACKED"), "1"));
    ggml_cpu_init();
    auto * backend = ggml_backend_cpu_init();
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    assert(backend && ctx);
    constexpr int k = 2880, m = 2880, experts = 32;
    auto * serial = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, k, m, experts);
    auto * parallel = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, k, m, experts);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_repack_buffer_type());
    assert(buffer && serial->extra && parallel->extra);
    const size_t bytes = ggml_nbytes(serial), stride = ggml_row_size(GGML_TYPE_MXFP4, k);
    std::vector<unsigned char> raw(bytes);
    uint32_t state = 71393;
    for (auto & x : raw) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        x = state;
    }
    const int order[] = {0, 1, 1, 0, 0, 1};
    bool ready[2] = {false, false};
    for (int index = 0; index < 6; ++index) {
        const int mode = order[index];
        setenv("GGML_CPU_MXFP4_PARALLEL_PACK", mode ? "1" : "0", 1);
        auto * dst = mode ? parallel : serial;
        const auto begin = std::chrono::steady_clock::now();
        const auto cpu_begin = std::clock();
        ggml_backend_tensor_set(dst, raw.data(), 0, bytes);
        const double cpu = double(std::clock() - cpu_begin) / CLOCKS_PER_SEC;
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        ready[mode] = true;
        if (ready[0] && ready[1]) assert(!std::memcmp(serial->data, parallel->data, bytes));
        std::printf("{\"mode\":\"%s\",\"index\":%d,\"bytes\":%zu,\"seconds\":%.9f,\"cpu_seconds\":%.9f,\"cpu_to_wall\":%.4f,\"max_threads\":%d}\n", mode ? "parallel" : "serial", index, bytes, elapsed, cpu, cpu / elapsed, omp_get_max_threads());
        std::fflush(stdout);
    }
    const auto * packed = (const unsigned char *) parallel->data;
    for (int row = 0; row < m * experts; ++row) {
        for (int block = 0; block < k / 32; ++block) {
            const auto * src = raw.data() + size_t(row) * stride + block * 17;
            const auto * dst = packed + size_t(row / 4) * 4 * stride + size_t(block) * 68;
            assert(dst[row % 4] == src[0]);
            assert(!std::memcmp(dst + 4 + (row % 4) * 16, src + 1, 16));
        }
    }
    std::printf("{\"passed\":true,\"bytes_roundtrip_checked\":%zu,\"serial_parallel_bit_equal\":true}\n", bytes);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

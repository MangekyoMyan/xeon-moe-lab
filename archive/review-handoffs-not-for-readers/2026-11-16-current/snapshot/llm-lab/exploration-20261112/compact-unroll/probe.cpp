#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <omp.h>
#include "compact-kernel.inc"
#include "iq-small-kernels.inc"
#include "variants.inc"

static size_t checked = 0;
static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> values) { std::sort(values.begin(), values.end()); return values[values.size()/2]; }

template<int type> static void run(int k, int rows, int threads, bool bench) {
    using block = typename std::conditional<type == 0, compact_block, iq_compact_block>::type;
    using dot = float (*)(int, const block *, const block_q8_K *);
    std::array<dot, 5> funcs;
    if constexpr(type == 0) funcs = {compact_dot, compact_dot_u1, compact_dot_u2, compact_dot_u4, compact_dot_u8};
    else if constexpr(type == 1) funcs = {iq_compact_dot<true>, iq_compact_dot_u1<true>, iq_compact_dot_u2<true>, iq_compact_dot_u4<true>, iq_compact_dot_u8<true>};
    else funcs = {iq_compact_dot<false>, iq_compact_dot_u1<false>, iq_compact_dot_u2<false>, iq_compact_dot_u4<false>, iq_compact_dot_u8<false>};
    const int blocks = k / 256;
    std::mt19937 rng(3917 + k + rows + type);
    std::vector<block> weights(size_t(rows) * blocks);
    std::vector<block_q8_K> input(blocks);
    std::vector<float> reference(rows), output(rows);
    for (auto & w : weights) {
        for (auto & q : w.qs) {
            if constexpr(type == 1) q = (rng()%3 + (rng()%2)*8) | ((rng()%3 + (rng()%2)*8) << 4);
            else q = rng();
        }
        w.d = float(int(rng()%1024) - 512) / 4096;
        if constexpr(type == 0) for (auto & s : w.scale) s = 2*(rng()%16) + 1;
        else for (auto & s : w.scales) s = rng();
    }
    for (auto & x : input) {
        x.d = float(int(rng()%1024) - 512) / 4096;
        for (auto & q : x.qs) q = int(rng()%256) - 128;
    }
    auto compute = [&](int mode, std::vector<float> & out) {
        const dot fn = funcs[mode];
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int row = 0; row < rows; ++row) out[row] = fn(k, weights.data() + size_t(row)*blocks, input.data());
    };
    compute(0, reference);
    for (int mode = 1; mode <= 4; ++mode) {
        compute(mode, output);
        if (std::memcmp(reference.data(), output.data(), size_t(rows)*4)) throw std::runtime_error("unroll bit mismatch");
        checked += rows;
    }
    if (!bench) return;
    for (int mode = 1; mode <= 4; ++mode) {
        std::vector<double> base, candidate;
        for (int rep = 0; rep < 4; ++rep) for (int arm : {0, mode, mode, 0}) {
            const double start = now();
            compute(arm, output);
            const double elapsed = now() - start;
            if (std::memcmp(reference.data(), output.data(), size_t(rows)*4)) throw std::runtime_error("timed bit mismatch");
            checked += rows;
            (arm ? candidate : base).push_back(elapsed);
        }
        std::printf("{\"type\":%d,\"k\":%d,\"rows\":%d,\"threads\":%d,\"unroll\":%d,\"base_ms\":%.6f,\"candidate_ms\":%.6f,\"speedup\":%.6f}\n",
            type, k, rows, threads, 1 << (mode-1), median(base), median(candidate), median(base)/median(candidate));
        std::fflush(stdout);
    }
}

int main(int argc, char **) try {
    for (int k : {256, 512, 2304, 5120}) for (int m : {1, 7, 129}) {
        run<0>(k, m, 1, false); run<1>(k, m, 1, false); run<2>(k, m, 1, false);
    }
    if (argc > 1) for (int threads : {1, 12}) for (int k : {2304, 5120}) {
        const int rows = k == 2304 ? 98304 : 49152;
        run<0>(k, rows, threads, true); run<1>(k, rows, threads, true); run<2>(k, rows, threads, true);
    }
    std::printf("{\"pass\":true,\"outputs_checked\":%zu}\n", checked);
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
}

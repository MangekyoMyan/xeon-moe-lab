#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

struct matrix {
    ggml_type type;
    const ggml_type_traits_cpu * cpu;
    size_t stride;
    size_t bytes;
    char * weights;
    std::vector<unsigned char> rhs;
    double error;
    double quant_rmse;
};

struct alignas(64) sink_value { double value; };
static sink_value sinks[24];

static void pin(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask)) { perror("affinity"); std::abort(); }
}

static float random_value(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return (int(state & 65535) - 32768) / 16384.0f;
}

static void unpack(ggml_type type, const void * data, float * out, int k) {
    if (type == GGML_TYPE_Q8_K) {
        dequantize_row_q8_K((const block_q8_K *) data, out, k);
    } else if (type == GGML_TYPE_Q8_1) {
        const auto * blocks = (const block_q8_1 *) data;
        for (int i = 0; i < k; ++i) {
            out[i] = ggml_fp16_to_fp32(blocks[i / 32].d) * blocks[i / 32].qs[i % 32];
        }
    } else {
        const auto * traits = ggml_get_type_traits(type);
        assert(traits->to_float);
        traits->to_float(data, out, k);
    }
}

static double measure(const matrix & a, int k, int m, int threads, int iterations) {
    double start = 0, finish = 0;
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        pin(tid);
        const int begin = int(int64_t(m) * tid / threads);
        const int end = int(int64_t(m) * (tid + 1) / threads);
        double sum = 0;
        #pragma omp barrier
        #pragma omp single
        start = omp_get_wtime();
        for (int it = 0; it < iterations; ++it) {
            for (int r = begin; r < end; ++r) {
                float dot;
                a.cpu->vec_dot(k, &dot, 0, a.weights + size_t(r) * a.stride, 0, a.rhs.data(), 0, 1);
                sum += dot;
            }
        }
        sinks[tid].value = sum;
        #pragma omp barrier
        #pragma omp single
        finish = omp_get_wtime();
    }
    for (int t = 0; t < threads; ++t) assert(std::isfinite(sinks[t].value));
    return finish - start;
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 7) return 2;
    const int k = std::stoi(argv[1]), m = std::stoi(argv[2]), threads = std::stoi(argv[3]);
    const double seconds = std::stod(argv[4]);
    const int rounds = std::stoi(argv[5]);
    const std::string only = argc == 7 ? argv[6] : "all";
    if (k < 32 || k > 16384 || k % 32 || m < 24 || m > 262144 ||
        threads < 1 || threads > 24 || seconds < 0.02 || seconds > 30 || rounds < 1 || rounds > 30) return 2;
    ggml_cpu_init();
    omp_set_dynamic(0);
    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_Q4_0,
        GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL};
    const int patterns = std::min(m, 256);
    std::vector<float> data(size_t(patterns) * k), rhs(k), dx(k), dy(k);
    uint32_t state = 0x17352;
    for (float & x : data) x = random_value(state);
    for (float & x : rhs) x = random_value(state);
    std::vector<matrix> matrices;
    for (ggml_type type : types) {
        if (only != "all" && only != ggml_type_name(type)) continue;
        if (k % ggml_blck_size(type)) {
            printf("{\"skipped\":\"%s\",\"k\":%d,\"reason\":\"block size does not divide row\"}\n", ggml_type_name(type), k);
            continue;
        }
        const auto * cpu = ggml_get_type_traits_cpu(type);
        assert(cpu->from_float && cpu->vec_dot);
        ggml_quantize_init(type);
        matrix a{};
        a.type = type; a.cpu = cpu;
        a.stride = ggml_row_size(type, k); a.bytes = a.stride * m;
        if (a.bytes > size_t(1) << 30) return 2;
        a.weights = (char *) mmap(nullptr, a.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(a.weights != MAP_FAILED);
        assert(madvise(a.weights, a.bytes, MADV_NOHUGEPAGE) == 0);
        a.rhs.resize(ggml_row_size(cpu->vec_dot_type, k));
        ggml_get_type_traits_cpu(cpu->vec_dot_type)->from_float(rhs.data(), a.rhs.data(), k);
        std::vector<unsigned char> quantized(size_t(patterns) * a.stride);
        unpack(cpu->vec_dot_type, a.rhs.data(), dy.data(), k);
        double squared = 0;
        for (int r = 0; r < patterns; ++r) {
            auto * row = quantized.data() + size_t(r) * a.stride;
            cpu->from_float(data.data() + size_t(r) * k, row, k);
            unpack(type, row, dx.data(), k);
            double expected = 0, absolute = 0;
            for (int c = 0; c < k; ++c) {
                expected += double(dx[c]) * dy[c];
                absolute += std::abs(double(dx[c]) * dy[c]);
                squared += std::pow(double(dx[c]) - data[size_t(r) * k + c], 2);
            }
            float actual;
            cpu->vec_dot(k, &actual, 0, row, 0, a.rhs.data(), 0, 1);
            const double error = std::abs(actual - expected);
            if (!std::isfinite(actual) || error > 1e-5 * absolute + 1e-3) {
                fprintf(stderr, "dot validation failed: %s row %d error %.9g\n", ggml_type_name(type), r, error);
                return 3;
            }
            a.error = std::max(a.error, error);
        }
        a.quant_rmse = std::sqrt(squared / (size_t(patterns) * k));
        #pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            pin(tid);
            const int begin = int(int64_t(m) * tid / threads);
            const int end = int(int64_t(m) * (tid + 1) / threads);
            for (int r = begin; r < end; ++r) {
                memcpy(a.weights + size_t(r) * a.stride, quantized.data() + size_t(r % patterns) * a.stride, a.stride);
            }
        }
        printf("{\"prepared\":\"%s\",\"rhs_type\":\"%s\",\"k\":%d,\"m\":%d,\"threads\":%d,\"bytes\":%zu,\"rhs_bytes\":%zu,\"checked_rows\":%d,\"dot_max_abs_error\":%.9g,\"weight_rmse\":%.9g}\n",
               ggml_type_name(type), ggml_type_name(cpu->vec_dot_type), k, m, threads, a.bytes, a.rhs.size(), patterns, a.error, a.quant_rmse);
        fflush(stdout);
        matrices.push_back(std::move(a));
    }
    if (matrices.empty()) return 2;
    std::vector<int> repeats(matrices.size());
    for (size_t i = 0; i < matrices.size(); ++i) {
        const int initial = m < 1024 ? 1000 : 3;
        measure(matrices[i], k, m, threads, initial);
        const double sample = measure(matrices[i], k, m, threads, initial);
        repeats[i] = std::clamp(int(std::ceil(seconds * initial / sample)), 1, 10000000);
    }
    printf("{\"ready\":true,\"pid\":%d}\n", getpid()); fflush(stdout);
    for (int round = 0; round < rounds; ++round) {
        for (size_t order = 0; order < matrices.size(); ++order) {
            const size_t index = round % 2 ? matrices.size() - 1 - order : order;
            const auto & a = matrices[index];
            const double elapsed = measure(a, k, m, threads, repeats[index]);
            printf("{\"type\":\"%s\",\"round\":%d,\"iterations\":%d,\"elapsed_s\":%.9f,\"seconds_per_matvec\":%.12f,\"weight_GB_s\":%.9f,\"Gweights_s\":%.9f}\n",
                   ggml_type_name(a.type), round, repeats[index], elapsed, elapsed / repeats[index],
                   a.bytes * double(repeats[index]) / elapsed / 1e9,
                   double(k) * m * repeats[index] / elapsed / 1e9);
            fflush(stdout);
        }
    }
    for (auto & a : matrices) munmap(a.weights, a.bytes);
    return 0;
}

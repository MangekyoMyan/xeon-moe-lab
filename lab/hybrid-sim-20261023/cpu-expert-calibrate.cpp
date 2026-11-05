#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>

static constexpr int H = 5120, F = 2304, POOL = 64;
static constexpr size_t BYTES = size_t(3)*H*F*17/32;
static const ggml_type_traits_cpu * dot;
static char * weights;
static std::vector<unsigned char> xh, xf;
struct alignas(64) Sink { double value; };
static Sink sinks[24];

static void pin(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
}

static double measure(int experts, int threads, int repeats) {
    const size_t stride_h = ggml_row_size(GGML_TYPE_MXFP4, H);
    const size_t stride_f = ggml_row_size(GGML_TYPE_MXFP4, F);
    double start = 0, finish = 0;
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        pin(tid);
        double sum = 0;
        #pragma omp barrier
        #pragma omp single
        start = omp_get_wtime();
        for (int it = 0; it < repeats; ++it) {
            // Consecutive layers use different physical weights; 64 experts exceed LLC.
            const int first = (it*7) % POOL;
            for (int phase = 0; phase < 3; ++phase) {
                const int k = phase < 2 ? H : F;
                const int rows = phase < 2 ? F : H;
                const size_t stride = phase < 2 ? stride_h : stride_f;
                const size_t offset = phase*stride_h*F;
                const auto & rhs = phase < 2 ? xh : xf;
                #pragma omp for schedule(static)
                for (int job = 0; job < experts*rows; ++job) {
                    const int e = (first + job/rows) % POOL;
                    float value;
                    dot->vec_dot(k, &value, 0, weights + e*BYTES + offset + (job%rows)*stride,
                                 0, rhs.data(), 0, 1);
                    sum += value;
                }
            }
        }
        sinks[tid].value = sum;
        #pragma omp barrier
        #pragma omp single
        finish = omp_get_wtime();
    }
    for (int i = 0; i < threads; ++i) assert(std::isfinite(sinks[i].value));
    return finish - start;
}

int main() {
    ggml_cpu_init(); omp_set_dynamic(0);
    dot = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const auto * rhs = ggml_get_type_traits_cpu(dot->vec_dot_type);
    assert(dot->vec_dot_type == GGML_TYPE_Q8_0);
    weights = (char *)mmap(nullptr, BYTES*POOL, PROT_READ|PROT_WRITE,
                           MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    assert(weights != MAP_FAILED);
    assert(madvise(weights, BYTES*POOL, MADV_NOHUGEPAGE) == 0);
    std::vector<float> data(H), unpacked(H), input(H), rhs_unpacked(H);
    for (int i = 0; i < H; ++i) input[i] = std::sin(i*0.173f);
    xh.resize(ggml_row_size(dot->vec_dot_type, H));
    xf.resize(ggml_row_size(dot->vec_dot_type, F));
    rhs->from_float(input.data(), xh.data(), H);
    rhs->from_float(input.data(), xf.data(), F);
    std::vector<unsigned char> patterns[2];
    for (int shape = 0; shape < 2; ++shape) {
        const int k = shape == 0 ? H : F;
        const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, k);
        patterns[shape].resize(16*stride);
        const auto & v = shape == 0 ? xh : xf;
        ggml_get_type_traits(dot->vec_dot_type)->to_float(v.data(), rhs_unpacked.data(), k);
        for (int pattern = 0; pattern < 16; ++pattern) {
            for (int i = 0; i < k; ++i) data[i] = std::sin((i+pattern*73)*0.371f)*std::ldexp(1.0f, pattern%8-6);
            auto * dst = patterns[shape].data() + pattern*stride;
            dot->from_float(data.data(), dst, k);
            ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(dst, unpacked.data(), k);
            double expected = 0, absolute = 0;
            for (int i = 0; i < k; ++i) {
                expected += double(unpacked[i])*rhs_unpacked[i];
                absolute += std::abs(double(unpacked[i])*rhs_unpacked[i]);
            }
            float actual;
            dot->vec_dot(k, &actual, 0, dst, 0, v.data(), 0, 1);
            assert(std::isfinite(actual) && std::abs(actual-expected) < 1e-5*absolute+1e-3);
        }
    }
    #pragma omp parallel num_threads(24)
    {
        pin(omp_get_thread_num());
        #pragma omp for schedule(static)
        for (int e = 0; e < POOL; ++e) {
            for (int phase = 0; phase < 3; ++phase) {
                const int shape = phase < 2 ? 0 : 1;
                const int k = shape == 0 ? H : F;
                const int rows = shape == 0 ? F : H;
                const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, k);
                for (int row = 0; row < rows; ++row)
                    std::memcpy(weights+e*BYTES+phase*(BYTES/3)+row*stride,
                                patterns[shape].data()+((row+e)%16)*stride,stride);
            }
        }
    }
    printf("{\"validation\":true,\"pool_bytes\":%zu,\"expert_bytes\":%zu}\n", BYTES*POOL, BYTES);
    fflush(stdout);
    for (int threads : {12,24}) {
        for (int round = 0; round < 5; ++round) {
            for (int index = 1; index <= 6; ++index) {
                const int count = round%2 ? 7-index : index;
                measure(count, threads, 16);
                const double sample = measure(count, threads, 16);
                const int repeats = std::clamp(int(std::ceil(0.3*16/sample)), 16, 10000);
                const double time = measure(count, threads, repeats);
                printf("{\"threads\":%d,\"round\":%d,\"experts\":%d,\"repeats\":%d,\"elapsed_s\":%.9f,\"ms_per_layer\":%.9f,\"logical_GB_s\":%.9f}\n",
                       threads,round,count,repeats,time,time*1000/repeats,BYTES*double(count)*repeats/time/1e9);
                fflush(stdout);
            }
        }
    }
    munmap(weights,BYTES*POOL);
}

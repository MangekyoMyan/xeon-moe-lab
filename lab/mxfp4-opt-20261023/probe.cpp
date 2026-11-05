#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
struct variant { std::string name; void * handle; dot_fn dot; int rows; };
struct alignas(64) sink_value { double value; };
static sink_value sinks[24];

static void pin(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
}

static uint32_t random_u32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill(block_mxfp4 * x, block_q8_0 * y, int nb, uint32_t state) {
    for (int b = 0; b < nb; ++b) {
        x[b].e = 115 + random_u32(state) % 20;
        for (int j = 0; j < 16; ++j) x[b].qs[j] = random_u32(state);
        y[b].d = ggml_fp32_to_fp16(std::ldexp(float(1 + random_u32(state) % 100), -12));
        for (int j = 0; j < 32; ++j) y[b].qs[j] = int(random_u32(state) % 255) - 127;
    }
}

static void check(const std::vector<variant> & vars, int k, const block_mxfp4 * x, const block_q8_0 * y, bool independent) {
    float ref;
    ggml_vec_dot_mxfp4_q8_0(k, &ref, 0, x, 0, y, 0, 1);
    for (const auto & v : vars) {
        float got;
        v.dot(k, &got, 0, x, 0, y, 0, 1);
        if (std::memcmp(&ref, &got, sizeof(float)) && !(std::isnan(ref) && std::isnan(got))) {
            std::fprintf(stderr, "Mismatch %s K%d %.9g %.9g\n", v.name.c_str(), k, ref, got);
            std::exit(3);
        }
        if (v.rows > 1) {
            std::vector<block_mxfp4> grouped(size_t(k / 32) * v.rows);
            std::vector<float> outputs(v.rows);
            const size_t stride = size_t(k / 32) * sizeof(block_mxfp4);
            for (int r = 0; r < v.rows; ++r) {
                auto * row = grouped.data() + size_t(r) * (k / 32);
                std::memcpy(row, x, stride);
                for (int b = 0; b < k / 32; ++b) row[b].qs[0] ^= r * 37;
            }
            v.dot(k, outputs.data(), 0, grouped.data(), stride, y, 0, v.rows);
            for (int r = 0; r < v.rows; ++r) {
                float expected;
                ggml_vec_dot_mxfp4_q8_0(k, &expected, 0, grouped.data() + size_t(r) * (k / 32), 0, y, 0, 1);
                if (std::memcmp(&expected, &outputs[r], sizeof(float)) && !(std::isnan(expected) && std::isnan(outputs[r]))) {
                    std::fprintf(stderr, "Grouped mismatch %s K%d row%d %.9g %.9g\n", v.name.c_str(), k, r, expected, outputs[r]);
                    std::exit(5);
                }
            }
        }
    }
    if (independent) {
        std::vector<float> dx(k), dy(k);
        dequantize_row_mxfp4(x, dx.data(), k);
        dequantize_row_q8_0(y, dy.data(), k);
        double expected = 0, absolute = 0;
        for (int i = 0; i < k; ++i) {
            const double p = double(dx[i]) * dy[i];
            expected += p;
            absolute += std::abs(p);
        }
        if (!std::isfinite(ref) || std::abs(double(ref) - expected) > 2e-6 * absolute + 1e-20) {
            std::fprintf(stderr, "Reference mismatch K%d %.12g %.12g bound %.12g\n", k, ref, expected, absolute);
            std::exit(4);
        }
    }
}

static void validate(const std::vector<variant> & vars) {
    const size_t page = sysconf(_SC_PAGESIZE);
    const int blocks[] = {1, 2, 3, 4, 5, 7, 8, 11, 16, 31, 32, 33, 64, 65, 71, 72, 90, 127, 128, 159, 160, 255, 256, 511, 512};
    uint64_t guarded = 0, halves = 0, exponents = 0;
    for (int nb : blocks) {
        const size_t xb = nb * sizeof(block_mxfp4), yb = nb * sizeof(block_q8_0);
        const size_t xp = (xb + page - 1) / page * page, yp = (yb + page - 1) / page * page;
        char * xm = (char *) mmap(nullptr, xp + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        char * ym = (char *) mmap(nullptr, yp + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(xm != MAP_FAILED && ym != MAP_FAILED);
        assert(!mprotect(xm + xp, page, PROT_NONE) && !mprotect(ym + yp, page, PROT_NONE));
        auto * x = (block_mxfp4 *) (xm + xp - xb);
        auto * y = (block_q8_0 *) (ym + yp - yb);
        for (uint32_t seed = 1; seed <= 128; ++seed) {
            fill(x, y, nb, seed * 7919);
            check(vars, nb * 32, x, y, true);
            ++guarded;
        }
        munmap(xm, xp + page);
        munmap(ym, yp + page);
    }
    block_mxfp4 x[3];
    block_q8_0 y[3];
    fill(x, y, 3, 197);
    for (int b = 0; b < 3; ++b) x[b].e = 126;
    for (uint32_t half = 0; half < 65536; ++half) {
        if ((half & 0x7c00) == 0x7c00) continue;
        for (int b = 0; b < 3; ++b) y[b].d = half;
        check(vars, 96, x, y, true);
        ++halves;
    }
    for (int e = 0; e <= 254; ++e) {
        for (int b = 0; b < 3; ++b) { x[b].e = e; y[b].d = 0x0400; }
        check(vars, 96, x, y, false);
        ++exponents;
    }
    uint64_t grouped_guards = 0;
    for (const auto & v : vars) {
        if (v.rows == 1) continue;
        for (int nb : {0, 1, 2, 3, 5, 71, 72, 90, 160}) {
            for (size_t padding : {size_t(0), size_t(7)}) {
                const size_t xb = nb * sizeof(block_mxfp4), yb = nb * sizeof(block_q8_0);
                const size_t stride = xb + padding, span = (v.rows - 1) * stride + xb;
                const size_t xp = (span + page - 1) / page * page, yp = (yb + page - 1) / page * page;
                char * xm = (char *) mmap(nullptr, xp + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                char * ym = (char *) mmap(nullptr, yp + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                assert(xm != MAP_FAILED && ym != MAP_FAILED);
                assert(!mprotect(xm + xp, page, PROT_NONE) && !mprotect(ym + yp, page, PROT_NONE));
                char * base = xm + xp - span;
                auto * rhs = (block_q8_0 *) (ym + yp - yb);
                for (uint32_t seed = 1; seed <= 32; ++seed) {
                    for (int r = 0; r < v.rows; ++r) fill((block_mxfp4 *) (base + r * stride), rhs, nb, seed + r * 79);
                    float actual[4];
                    v.dot(nb * 32, actual, 0, base, stride, rhs, 0, v.rows);
                    for (int r = 0; r < v.rows; ++r) {
                        float expected;
                        ggml_vec_dot_mxfp4_q8_0(nb * 32, &expected, 0, base + r * stride, 0, rhs, 0, 1);
                        assert(!std::memcmp(&expected, &actual[r], sizeof(float)));
                    }
                    ++grouped_guards;
                }
                munmap(xm, xp + page);
                munmap(ym, yp + page);
            }
        }
    }
    std::printf("{\"guarded_group_inputs\":%llu,\"zero_length_and_odd_stride_tested\":true}\n", (unsigned long long) grouped_guards);
    std::printf("{\"validated\":true,\"guarded_inputs\":%llu,\"finite_half_inputs\":%llu,\"exponent_inputs\":%llu,\"variants\":%zu,\"bit_equal\":true,\"rhs_range\":[-127,127]}\n",
        (unsigned long long) guarded, (unsigned long long) halves, (unsigned long long) exponents, vars.size());
    std::fflush(stdout);
}

static double measure(const variant & v, const char * x, const void * y, size_t stride, int k, int m, int threads, int iterations) {
    double begin = 0, end = 0;
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        pin(tid);
        const int r0 = int(int64_t(m) * tid / threads), r1 = int(int64_t(m) * (tid + 1) / threads);
        double sum = 0;
        #pragma omp barrier
        #pragma omp single
        begin = omp_get_wtime();
        for (int it = 0; it < iterations; ++it) {
            int r = r0;
            for (; v.rows > 1 && r + v.rows <= r1; r += v.rows) {
                float values[4];
                v.dot(k, values, 0, x + size_t(r) * stride, stride, y, 0, v.rows);
                for (int j = 0; j < v.rows; ++j) sum += values[j];
            }
            for (; r < r1; ++r) {
                float value;
                v.dot(k, &value, 0, x + size_t(r) * stride, 0, y, 0, 1);
                sum += value;
            }
        }
        sinks[tid].value = sum;
        #pragma omp barrier
        #pragma omp single
        end = omp_get_wtime();
    }
    for (int t = 0; t < threads; ++t) assert(std::isfinite(sinks[t].value));
    return end - begin;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 8) return 2;
    ggml_cpu_init();
    omp_set_dynamic(0);
    std::vector<variant> vars;
    std::string list = argv[2];
    while (!list.empty()) {
        const auto pos = list.find(',');
        const auto name = list.substr(0, pos);
        assert(name.find('/') == std::string::npos);
        const std::string file = std::string(argv[1]) + '/' + name + ".so";
        void * handle = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
        const int rows = name.find("rows2") == 0 ? 2 : (name.find("rows4") == 0 ? 4 : 1);
        auto dot = (dot_fn) dlsym(handle, rows > 1 ? "ggml_vec_dot_mxfp4_q8_0_grouped" : "ggml_vec_dot_mxfp4_q8_0");
        assert(dot);
        vars.push_back({name, handle, dot, rows});
        if (pos == std::string::npos) break;
        list.erase(0, pos + 1);
    }
    if (argc == 4 && std::string(argv[3]) == "validate") {
        validate(vars);
    } else {
        if (argc != 8) return 2;
        const int k = std::stoi(argv[3]), m = std::stoi(argv[4]), threads = std::stoi(argv[5]), rounds = std::stoi(argv[7]);
        const double seconds = std::stod(argv[6]);
        if (k < 32 || k > 32768 || k % 32 || m < 1 || m > 262144 || threads < 1 || threads > 24 || seconds < .01 || seconds > 30 || rounds < 1 || rounds > 30) return 2;
        const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, k), bytes = stride * m;
        if (bytes > size_t(2) * 1024 * 1024 * 1024) return 2;
        auto * x = (char *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(x != MAP_FAILED && !madvise(x, bytes, MADV_NOHUGEPAGE));
        std::vector<block_q8_0> y(k / 32);
        const int patterns = std::min(m, 256);
        std::vector<block_mxfp4> pattern(size_t(patterns) * (k / 32));
        std::vector<block_q8_0> unused(k / 32);
        for (int r = 0; r < patterns; ++r) fill(pattern.data() + size_t(r) * (k / 32), unused.data(), k / 32, r + 1);
        fill(pattern.data(), y.data(), k / 32, 779);
        for (int r = 0; r < patterns; ++r) check(vars, k, pattern.data() + size_t(r) * (k / 32), y.data(), true);
        #pragma omp parallel num_threads(threads)
        {
            pin(omp_get_thread_num());
            #pragma omp for schedule(static)
            for (int r = 0; r < m; ++r) std::memcpy(x + size_t(r) * stride, pattern.data() + size_t(r % patterns) * (k / 32), stride);
        }
        std::vector<int> iterations;
        for (const auto & v : vars) {
            int initial = m < 1024 ? 500 : 3;
            measure(v, x, y.data(), stride, k, m, threads, initial);
            double sample = measure(v, x, y.data(), stride, k, m, threads, initial);
            iterations.push_back(std::clamp(int(seconds * initial / sample), 1, 10000000));
        }
        std::printf("{\"prepared\":true,\"k\":%d,\"m\":%d,\"threads\":%d,\"bytes\":%zu,\"bit_equal\":true}\n", k, m, threads, bytes);
        for (int round = 0; round < rounds; ++round) {
            for (size_t order = 0; order < vars.size(); ++order) {
                const size_t vi = round % 2 ? vars.size() - 1 - order : order;
                const auto & v = vars[vi];
                double elapsed = measure(v, x, y.data(), stride, k, m, threads, iterations[vi]);
                std::printf("{\"variant\":\"%s\",\"round\":%d,\"iterations\":%d,\"elapsed_s\":%.9f,\"ms_per_matvec\":%.9f,\"weight_GBs\":%.6f,\"GOPS\":%.6f}\n",
                    v.name.c_str(), round, iterations[vi], elapsed, elapsed * 1000 / iterations[vi], double(bytes) * iterations[vi] / elapsed / 1e9, double(k) * m * 2 * iterations[vi] / elapsed / 1e9);
                std::fflush(stdout);
            }
        }
        munmap(x, bytes);
    }
    for (const auto & v : vars) dlclose(v.handle);
}

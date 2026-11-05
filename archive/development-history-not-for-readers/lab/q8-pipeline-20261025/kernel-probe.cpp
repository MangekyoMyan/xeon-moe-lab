#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <linux/mempolicy.h>
#include <omp.h>
#include <sched.h>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
struct variant { std::string name; dot_fn dot; };
static uint32_t state = 20261025;
static uint32_t random_u32() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
static void fill(block_q8_0 * ptr, size_t count) {
    for (size_t b = 0; b < count; ++b) {
        ptr[b].d = ggml_fp32_to_fp16(float(1 + random_u32() % 8191) / 262144.0f);
        for (int i = 0; i < 32; ++i) ptr[b].qs[i] = int(random_u32() % 256) - 128;
    }
}
static std::vector<std::string> split(const std::string & text) {
    std::vector<std::string> result;
    size_t begin = 0;
    do {
        size_t end = text.find(',', begin);
        result.push_back(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    } while (true);
    return result;
}
static void exact(dot_fn reference, const std::vector<variant> & variants, int k, const void * x, const void * y, size_t & count) {
    float expected;
    reference(k, &expected, 0, x, 0, y, 0, 1);
    for (const auto & item : variants) {
        float output[] = {17, 0, 19};
        item.dot(k, output + 1, 0, x, 0, y, 0, 1);
        if (output[0] != 17 || output[2] != 19 || std::memcmp(&expected, output + 1, sizeof(float))) {
            std::fprintf(stderr, "Mismatch variant=%s k=%d expected=%a got=%a\n", item.name.c_str(), k, expected, output[1]);
            std::exit(3);
        }
        ++count;
    }
}
int main(int argc, char ** argv) {
    if (argc != 4 && argc != 10) return 2;
    ggml_cpu_init();
    void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
    auto reference = reinterpret_cast<dot_fn>(dlsym(library, "ggml_vec_dot_q8_0_q8_0"));
    assert(reference);
    std::vector<variant> variants;
    for (const auto & name : split(argv[2])) {
        const std::string symbol = name == "baseline" ? "ggml_vec_dot_q8_0_q8_0" : "ivb_q8_" + name;
        auto fn = reinterpret_cast<dot_fn>(dlsym(library, symbol.c_str()));
        if (!fn) { std::fprintf(stderr, "Missing %s\n", symbol.c_str()); return 2; }
        variants.push_back({name, fn});
    }
    if (std::strcmp(argv[3], "validate") == 0) {
        const size_t page = sysconf(_SC_PAGESIZE);
        size_t comparisons = 0;
        for (int k : {0, 32, 64, 96, 128, 160, 224, 288, 1408, 4096, 10944, 12288, 16384}) {
            const size_t bytes = size_t(k / 32) * sizeof(block_q8_0);
            const size_t span = ((bytes + page - 1) / page + 1) * page;
            auto * mapping = static_cast<char *>(mmap(nullptr, 2 * span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            assert(mapping != MAP_FAILED);
            assert(mprotect(mapping + span - page, page, PROT_NONE) == 0);
            assert(mprotect(mapping + 2 * span - page, page, PROT_NONE) == 0);
            auto * x = reinterpret_cast<block_q8_0 *>(mapping + span - page - bytes);
            auto * y = reinterpret_cast<block_q8_0 *>(mapping + 2 * span - page - bytes);
            for (int trial = 0; trial < 128; ++trial) {
                fill(x, k / 32); fill(y, k / 32);
                exact(reference, variants, k, x, y, comparisons);
            }
            assert(munmap(mapping, 2 * span) == 0);
        }
        block_q8_0 x[2], y[2];
        fill(x, 2); fill(y, 2);
        size_t finite_scales = 0;
        for (unsigned half = 0; half <= 65535; ++half) {
            if ((half & 0x7c00) == 0x7c00) continue;
            x[0].d = y[1].d = half;
            y[0].d = x[1].d = 0x3c00;
            exact(reference, variants, 64, x, y, comparisons);
            ++finite_scales;
        }
        std::printf("{\"validated\":true,\"variants\":%zu,\"exact_comparisons\":%zu,\"finite_half_patterns\":%zu}\n", variants.size(), comparisons, finite_scales);
        return 0;
    }
    if (std::strcmp(argv[3], "bench") != 0) return 2;
    const int k = std::stoi(argv[4]), m = std::stoi(argv[5]), threads = std::stoi(argv[6]);
    const double seconds = std::stod(argv[7]);
    const int rounds = std::stoi(argv[8]), first_cpu = std::stoi(argv[9]);
    assert(k >= 32 && k <= 16384 && k % 32 == 0 && m > 0 && m <= 262144 && threads > 0 && threads <= 24);
    assert(first_cpu >= 0 && first_cpu + threads <= 24 && seconds > 0 && seconds <= 10 && rounds > 0 && rounds <= 50);
    const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, k), bytes = stride * m;
    assert(bytes <= size_t(1) << 30);
    auto * x = static_cast<char *>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    assert(x != MAP_FAILED);
    assert(madvise(x, bytes, MADV_NOHUGEPAGE) == 0);
    const char * layout_env = std::getenv("IVB_PROBE_LAYOUT");
    const std::string layout = layout_env ? layout_env : "inherited";
    const char * order_env = std::getenv("IVB_PROBE_ORDER");
    const bool socket_order = order_env && std::strcmp(order_env, "socket") == 0;
    if (layout != "inherited") {
        const size_t page = sysconf(_SC_PAGESIZE);
        const size_t half = bytes / 2 / page * page;
        unsigned long mask = layout == "all1" ? 2 : 1;
        if (layout == "interleave") mask = 3;
        const int mode = layout == "interleave" ? MPOL_INTERLEAVE : MPOL_BIND;
        assert(syscall(SYS_mbind, x, bytes, mode, &mask, sizeof(mask) * 8, 0) == 0);
        if (layout == "rowlocal" || layout == "opposite") {
            assert(first_cpu == 0 && (threads == 12 || threads == 24) && socket_order);
            mask = layout == "opposite" ? 2 : 1;
            assert(syscall(SYS_mbind, x, half, MPOL_BIND, &mask, sizeof(mask) * 8, 0) == 0);
            mask = layout == "opposite" ? 1 : 2;
            assert(syscall(SYS_mbind, x + half, bytes - half, MPOL_BIND, &mask, sizeof(mask) * 8, 0) == 0);
        }
    }
    std::vector<block_q8_0> y(k / 32);
    std::vector<float> expected(m), output(m);
    fill(reinterpret_cast<block_q8_0 *>(x), size_t(m) * k / 32); fill(y.data(), y.size());
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t page_count = (bytes + page_size - 1) / page_size;
    const size_t sample_count = std::min(size_t(512), page_count);
    std::vector<void *> page_addresses(sample_count);
    std::vector<int> page_nodes(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        page_addresses[i] = x + (page_count * i / sample_count) * page_size;
    }
    if (syscall(SYS_move_pages, 0, sample_count, page_addresses.data(), nullptr, page_nodes.data(), 0) != 0) {
        std::perror("move_pages query");
        return 4;
    }
    size_t nodes[2] = {};
    for (size_t i = 0; i < sample_count; ++i) {
        const int node = page_nodes[i];
        assert(node == 0 || node == 1);
        ++nodes[node];
        if (layout == "all0" || layout == "all1") assert(node == (layout == "all1"));
        if (layout == "rowlocal" || layout == "opposite") {
            const bool upper = static_cast<char *>(page_addresses[i]) - x >= ptrdiff_t(bytes / 2 / page_size * page_size);
            assert(node == int(upper ^ (layout == "opposite")));
        }
    }
    std::fprintf(stderr, "PLACEMENT layout=%s node0=%zu node1=%zu samples=%zu\n", layout.c_str(), nodes[0], nodes[1], sample_count);
    for (int row = 0; row < m; ++row) reference(k, &expected[row], 0, x + row * stride, 0, y.data(), 0, 1);
    omp_set_dynamic(0);
    for (int round = -1; round < rounds; ++round) {
        for (size_t order = 0; order < variants.size(); ++order) {
            const size_t index = round >= 0 && round % 2 ? variants.size() - 1 - order : order;
            const auto & item = variants[index];
            const double duration = round < 0 ? std::min(seconds, .03) : seconds;
            double start = 0, elapsed = 0;
            unsigned iterations = 0;
            bool again = true;
            #pragma omp parallel num_threads(threads) shared(start, elapsed, iterations, again)
            {
                const int tid = omp_get_thread_num();
                cpu_set_t mask;
                CPU_ZERO(&mask); CPU_SET(first_cpu + tid, &mask);
                assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
                int rank = tid;
                if (socket_order && threads == 24) {
                    assert(first_cpu == 0);
                    rank = tid % 6 + (tid >= 12 ? 6 : 0) + (tid % 12 >= 6 ? 12 : 0);
                }
                const int begin = int(int64_t(m) * rank / threads), end = int(int64_t(m) * (rank + 1) / threads);
                #pragma omp barrier
                #pragma omp master
                start = omp_get_wtime();
                #pragma omp barrier
                do {
                    for (int row = begin; row < end; ++row) item.dot(k, &output[row], 0, x + row * stride, 0, y.data(), 0, 1);
                    #pragma omp barrier
                    #pragma omp master
                    {
                        ++iterations;
                        elapsed = omp_get_wtime() - start;
                        again = elapsed < duration;
                    }
                    #pragma omp barrier
                } while (again);
            }
            if (std::memcmp(expected.data(), output.data(), m * sizeof(float))) return 3;
            if (round >= 0) {
                std::printf("{\"K\":%d,\"M\":%d,\"threads\":%d,\"first_cpu\":%d,\"round\":%d,\"variant\":\"%s\",\"layout\":\"%s\",\"socket_order\":%s,\"iterations\":%u,\"seconds\":%.9g,\"seconds_per_matrix\":%.9g,\"weight_GB_s\":%.9g}\n",
                    k, m, threads, first_cpu, round, item.name.c_str(), layout.c_str(), socket_order ? "true" : "false", iterations, elapsed, elapsed / iterations, double(bytes) * iterations / elapsed / 1e9);
                std::fflush(stdout);
            }
        }
    }
    munmap(x, bytes);
    dlclose(library);
}

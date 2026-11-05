#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "llamafile/sgemm.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
using cols_fn = void (*)(int, float *, const void *, const void * const *);
static uint32_t random_state = 0x20261025;
static uint32_t random_u32() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
static void fill(block_q8_0 * data, size_t blocks) {
    for (size_t b = 0; b < blocks; ++b) {
        uint16_t half;
        do { half = random_u32(); } while ((half & 0x7c00) == 0x7c00);
        data[b].d = half;
        for (int i = 0; i < 32; ++i) data[b].qs[i] = int(random_u32() % 256) - 128;
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_cpu_init();
    void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(library);
    auto dot = reinterpret_cast<dot_fn>(dlsym(library, "ggml_vec_dot_q8_0_q8_0"));
    auto moe_cols2 = reinterpret_cast<cols_fn>(dlsym(library, "ggml_vec_dot_q8_0_q8_0_cols2"));
    cols_fn dense[3];
    const int widths[] = {1, 2, 4};
    for (int i = 0; i < 3; ++i) {
        char symbol[80];
        std::snprintf(symbol, sizeof(symbol), "ggml_vec_dot_q8_0_q8_0_sgemm_cols%d", widths[i]);
        dense[i] = reinterpret_cast<cols_fn>(dlsym(library, symbol));
        assert(dense[i]);
    }
    assert(dot && moe_cols2);
    ggml_compute_params params{};
    params.nth = 1;
    size_t comparisons = 0;
    auto check = [&](int k, const block_q8_0 * x, const void * const y[4]) {
        const size_t blocks = k / 32;
        std::vector<block_q8_0> packed(4 * blocks + 1);
        for (int c = 0; c < 4; ++c) std::memcpy(packed.data() + c * blocks, y[c], blocks * sizeof(block_q8_0));
        float expected[4] = {};
        assert(llamafile_sgemm(&params, 1, 4, blocks, x, blocks, packed.data(), blocks,
                              expected, 1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32));
        for (int i = 0; i < 3; ++i) {
            float output[6] = {17, 0, 0, 0, 0, 19};
            const int width = widths[i];
            output[width + 1] = 19;
            dense[i](k, output + 1, x, y);
            if (output[0] != 17 || output[width + 1] != 19 || std::memcmp(expected, output + 1, width * sizeof(float))) {
                std::fprintf(stderr, "Dense mismatch k=%d cols=%d\n", k, width);
                std::exit(3);
            }
            comparisons += width;
        }
        float original[2], actual[] = {17, 0, 0, 19};
        for (int c = 0; c < 2; ++c) dot(k, original + c, 0, x, 0, y[c], 0, 1);
        moe_cols2(k, actual + 1, x, y);
        assert(actual[0] == 17 && actual[3] == 19 && !std::memcmp(original, actual + 1, sizeof(original)));
        comparisons += 2;
    };
    const size_t page = sysconf(_SC_PAGESIZE);
    for (int k : {0, 32, 64, 96, 224, 256, 288, 1408, 4096, 10944, 12288, 32768}) {
        const size_t blocks = k / 32, bytes = blocks * sizeof(block_q8_0);
        const size_t span = ((bytes + page - 1) / page + 1) * page;
        char * memory = static_cast<char *>(mmap(nullptr, span * 5, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        assert(memory != MAP_FAILED);
        block_q8_0 * buffers[5];
        for (int c = 0; c < 5; ++c) {
            assert(mprotect(memory + (c + 1) * span - page, page, PROT_NONE) == 0);
            buffers[c] = reinterpret_cast<block_q8_0 *>(memory + (c + 1) * span - page - bytes);
        }
        const void * y[4] = {buffers[1], buffers[2], buffers[3], buffers[4]};
        for (int trial = 0; trial < 64; ++trial) {
            for (auto * buffer : buffers) fill(buffer, blocks);
            check(k, buffers[0], y);
        }
        munmap(memory, span * 5);
    }
    block_q8_0 data[5][2];
    for (auto & row : data) fill(row, 2);
    const void * y[4] = {data[1], data[2], data[3], data[4]};
    size_t finite_halves = 0;
    for (unsigned half = 0; half < 65536; ++half) {
        if ((half & 0x7c00) == 0x7c00) continue;
        data[0][0].d = data[1][1].d = half;
        check(64, data[0], y);
        ++finite_halves;
    }
    std::printf("{\"passed\":true,\"compared_outputs\":%zu,\"finite_half_patterns\":%zu}\n", comparisons, finite_halves);
    dlclose(library);
}

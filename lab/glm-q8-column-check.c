#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*cols_fn)(int, float *, const void *, const void * const [4]);
static uint32_t state = 0x743917;

static uint32_t rng(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill(block_q8_0 * p, int nb) {
    for (int b = 0; b < nb; ++b) {
        uint16_t h;
        do { h = rng(); } while ((h & 0x7c00) == 0x7c00);
        p[b].d = h;
        for (int j = 0; j < 32; ++j) p[b].qs[j] = (int) (rng() % 256) - 128;
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_cpu_init();
    void * lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) return 2;
    dot_fn dot;
    cols_fn cols;
    *(void **) &dot = dlsym(lib, "ggml_vec_dot_q8_0_q8_0");
    *(void **) &cols = dlsym(lib, "ggml_vec_dot_q8_0_q8_0_cols4");
    if (!dot || !cols) return 2;
    const int lengths[] = {0, 32, 64, 96, 224, 256, 288, 1408, 4096, 10944, 12288, 32768};
    const size_t page = sysconf(_SC_PAGESIZE);
    int checked = 0;
    for (size_t a = 0; a < sizeof(lengths) / sizeof(lengths[0]); ++a) {
        const int n = lengths[a], nb = n / 32;
        const size_t bytes = (size_t) nb * sizeof(block_q8_0);
        const size_t span = ((bytes + page - 1) / page + 1) * page;
        char * raw = mmap(NULL, span * 5, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) return 2;
        const void * y[4];
        char * x = raw + span - page - bytes;
        for (int r = 0; r < 5; ++r) {
            if (mprotect(raw + (r + 1) * span - page, page, PROT_NONE)) return 2;
            if (r) y[r - 1] = x + r * span;
        }
        for (int trial = 0; trial < 64; ++trial) {
            for (int r = 0; r < 5; ++r) fill((block_q8_0 *) (x + r * span), nb);
            float reference[4], actual[6] = {17, 0, 0, 0, 0, 19};
            for (int c = 0; c < 4; ++c) {
                dot(n, reference + c, 0, x, 0, y[c], 0, 1);
                if (!isfinite(reference[c])) return 3;
            }
            cols(n, actual + 1, x, y);
            if (actual[0] != 17 || actual[5] != 19 || memcmp(reference, actual + 1, sizeof(reference))) {
                fprintf(stderr, "Mismatch n=%d trial=%d\n", n, trial);
                return 3;
            }
            checked += 4;
        }
        munmap(raw, span * 5);
    }
    block_q8_0 x[2], ys[4][2];
    const void * y[4] = {ys[0], ys[1], ys[2], ys[3]};
    fill(x, 2);
    for (int c = 0; c < 4; ++c) fill(ys[c], 2);
    int finite_halves = 0;
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        x[0].d = h;
        ys[0][1].d = h;
        float reference[4], actual[4];
        for (int c = 0; c < 4; ++c) dot(64, reference + c, 0, x, 0, y[c], 0, 1);
        cols(64, actual, x, y);
        if (memcmp(reference, actual, sizeof(reference))) return 3;
        ++finite_halves;
    }
    printf("{\"guarded_exact_outputs\":%d,\"finite_half_patterns\":%d,\"passed\":true}\n",
           checked, finite_halves);
    dlclose(lib);
    return 0;
}

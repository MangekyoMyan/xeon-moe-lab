#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <dlfcn.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*rows_fn)(int, float *, const void *, size_t, const void *);
static dot_fn single;
static rows_fn grouped[5];
static const int steps[] = {1, 2, 4, 2, 4};
static uint32_t state = 1234567;

static uint32_t rng(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill(block_q8_0 * p, size_t blocks) {
    for (size_t b = 0; b < blocks; ++b) {
        p[b].d = ggml_fp32_to_fp16(((int) (rng() % 8191) - 4095) / 65536.0f);
        for (int k = 0; k < 32; ++k) {
            p[b].qs[k] = (int) (rng() % 256) - 128;
        }
    }
}

static void compute(int variant, int k, int m, float * out, const char * x, size_t stride, const void * y) {
    int r = 0;
    const int step = steps[variant];
    if (variant) {
        for (; r + step <= m; r += step) {
            grouped[variant](k, out + r, x + r * stride, stride, y);
        }
    }
    for (; r < m; ++r) {
        single(k, out + r, 0, x + r * stride, 0, y, 0, 1);
    }
}

int main(int argc, char ** argv) {
    if (argc != 7) return 2;
    const int k = atoi(argv[2]), m = atoi(argv[3]), threads = atoi(argv[4]);
    const int rounds = atoi(argv[5]), iterations = atoi(argv[6]);
    if (k < 32 || k > 16384 || k % 32 || m < 1 || m > 262144 ||
        threads < 1 || threads > 24 || rounds < 1 || rounds > 100 ||
        iterations < 1 || iterations > 1000) return 2;
    const size_t stride = (size_t) k / 32 * sizeof(block_q8_0);
    if (stride * m > (size_t) 1 << 30) return 2;
    ggml_cpu_init();
    void * lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    *(void **) &single = dlsym(lib, "ggml_vec_dot_q8_0_q8_0");
    const char * names[] = {"single_pf8", "glm_q8_rows2_pf0", "ggml_vec_dot_q8_0_q8_0_rows4",
                           "glm_q8_rows2_pf8", "glm_q8_rows4_pf8"};
    if (!single) return 2;
    for (int v = 1; v < 5; ++v) {
        *(void **) &grouped[v] = dlsym(lib, names[v]);
        if (!grouped[v]) return 2;
    }
    const size_t page = sysconf(_SC_PAGESIZE);
    const int lengths[] = {32, 64, 96, 224, 256, 288, 1408, 4096, 10944, 12288};
    int guarded = 0;
    for (size_t a = 0; a < sizeof(lengths) / sizeof(lengths[0]); ++a) {
        const int n = lengths[a];
        const size_t bytes = (size_t) n / 32 * sizeof(block_q8_0);
        const size_t span = ((bytes + page - 1) / page + 1) * page;
        char * raw = mmap(NULL, span * 5, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) return 2;
        for (int r = 0; r < 5; ++r) {
            if (mprotect(raw + (r + 1) * span - page, page, PROT_NONE)) return 2;
        }
        char * x = raw + span - page - bytes;
        char * y = x + 4 * span;
        for (int trial = 0; trial < 32; ++trial) {
            for (int r = 0; r < 5; ++r) fill((block_q8_0 *) (x + r * span), n / 32);
            float ref[4];
            compute(0, n, 4, ref, x, span, y);
            for (int v = 1; v < 5; ++v) {
                float actual[6] = {17, 0, 0, 0, 0, 19};
                compute(v, n, 4, actual + 1, x, span, y);
                if (actual[0] != 17 || actual[5] != 19 || memcmp(ref, actual + 1, sizeof(ref))) return 3;
                guarded += 4;
            }
        }
        munmap(raw, span * 5);
    }
    char * x = malloc(stride * m);
    void * y = malloc(stride);
    float * out = malloc((size_t) m * sizeof(float));
    float * ref = malloc((size_t) m * sizeof(float));
    if (!x || !y || !out || !ref) return 2;
    fill((block_q8_0 *) x, (size_t) m * k / 32);
    fill(y, k / 32);
    compute(0, k, m, ref, x, stride, y);
    omp_set_dynamic(0);
    printf("{\"k\":%d,\"m\":%d,\"threads\":%d,\"guarded_exact_outputs\":%d,\"runs\":[",
           k, m, threads, guarded);
    for (int round = -1; round < rounds; ++round) {
        for (int order = 0; order < 5; ++order) {
            const int v = round >= 0 && round % 2 ? 4 - order : order;
            const double start = omp_get_wtime();
            #pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num();
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(tid, &mask);
                if (sched_setaffinity(0, sizeof(mask), &mask)) abort();
                const int begin = (int) ((int64_t) m * tid / threads);
                const int end = (int) ((int64_t) m * (tid + 1) / threads);
                for (int it = 0; it < iterations; ++it) {
                    compute(v, k, end - begin, out + begin, x + begin * stride, stride, y);
                    #pragma omp barrier
                }
            }
            const double seconds = (omp_get_wtime() - start) / iterations;
            if (memcmp(ref, out, (size_t) m * sizeof(float))) return 3;
            if (round >= 0) printf("%s{\"round\":%d,\"variant\":%d,\"seconds\":%.9f}",
                                   round || order ? "," : "", round, v, seconds);
        }
    }
    puts("]}");
    free(x); free(y); free(out); free(ref);
    dlclose(lib);
    return 0;
}

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

static void * allocate(size_t bytes) {
    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) exit(2);
    return ptr;
}

static void fill_row(void * row, float * values, int k, int seed) {
    for (int j = 0; j < k; ++j) {
        values[j] = (float) (((uint32_t) j * 1664525u + (uint32_t) seed * 1013904223u) % 65521u) / 32760.0f - 1.0f;
    }
    quantize_row_q8_0_ref(values, row, k);
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: probe LIBRARY_DIRECTORY K M THREADS ROUNDS\n");
        return 1;
    }
    const int k = atoi(argv[2]), m = atoi(argv[3]), threads = atoi(argv[4]), rounds = atoi(argv[5]);
    if (k < 32 || k > 32768 || k % 32 || m < 1 || m > 262144 ||
        threads < 1 || threads > 12 || rounds < 1 || rounds > 100) return 1;
    ggml_cpu_init();
    const int distances[] = {0, 8, 16, 32};
    dot_fn fn[4];
    void * handles[4];
    for (int v = 0; v < 4; ++v) {
        char path[4096];
        if (snprintf(path, sizeof(path), "%s/prefetch-%d.so", argv[1], distances[v]) >= (int) sizeof(path)) return 2;
        handles[v] = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handles[v]) { fprintf(stderr, "%s\n", dlerror()); return 2; }
        *(void **) &fn[v] = dlsym(handles[v], "ggml_vec_dot_q8_0_q8_0");
        if (!fn[v]) return 2;
    }
    // Put each short input immediately before an inaccessible page.
    const long page = sysconf(_SC_PAGESIZE);
    int cases = 0;
    for (int nb = 1; nb <= 65; ++nb) {
        const size_t bytes = ggml_row_size(GGML_TYPE_Q8_0, nb * 32);
        char * guarded = mmap(NULL, (size_t) page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (guarded == MAP_FAILED || bytes > (size_t) page || mprotect(guarded + page, page, PROT_NONE)) return 2;
        void * x = guarded + page - bytes;
        void * y = allocate(bytes);
        float * values = allocate(nb * 32 * sizeof(float));
        for (int seed = 0; seed < 8; ++seed) {
            fill_row(x, values, nb * 32, seed);
            fill_row(y, values, nb * 32, seed + 19);
            float ref;
            fn[0](nb * 32, &ref, 0, x, 0, y, 0, 1);
            if (!isfinite(ref)) return 3;
            for (int v = 1; v < 4; ++v) {
                float result;
                fn[v](nb * 32, &result, 0, x, 0, y, 0, 1);
                if (memcmp(&ref, &result, sizeof(float))) return 3;
                ++cases;
            }
        }
        munmap(guarded, page * 2);
        free(y);
        free(values);
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, k);
    char * x = allocate(row_bytes * (size_t) m);
    void * y = allocate(row_bytes);
    float * values = allocate(k * sizeof(float));
    float * result = allocate(m * sizeof(float));
    float * ref = allocate(m * sizeof(float));
    for (int i = 0; i < m; ++i) fill_row(x + i * row_bytes, values, k, i);
    fill_row(y, values, k, 77);
    omp_set_dynamic(0);
    printf("{\"k\":%d,\"m\":%d,\"threads\":%d,\"guarded_exact_cases\":%d,\"runs\":[", k, m, threads, cases);
    for (int round = -1; round < rounds; ++round) {
        for (int order = 0; order < 4; ++order) {
            const int v = round < 0 ? order : (round % 2 ? 3 - order : order);
            const double start = omp_get_wtime();
            #pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num();
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(tid, &mask);
                if (sched_setaffinity(0, sizeof(mask), &mask)) abort();
                #pragma omp for schedule(static)
                for (int i = 0; i < m; ++i) fn[v](k, result + i, 0, x + i * row_bytes, 0, y, 0, 1);
            }
            const double seconds = omp_get_wtime() - start;
            if (round == -1 && v == 0) memcpy(ref, result, m * sizeof(float));
            else if (memcmp(ref, result, m * sizeof(float))) return 3;
            if (round >= 0) printf("%s{\"round\":%d,\"distance\":%d,\"seconds\":%.9f}",
                round || order ? "," : "", round, distances[v], seconds);
        }
    }
    printf("]}\n");
    free(x); free(y); free(values); free(result); free(ref);
    for (int v = 0; v < 4; ++v) dlclose(handles[v]);
    return 0;
}

#define _GNU_SOURCE
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
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
static const char * names[] = {"original","expanded"};
#define VARIANTS ((int) (sizeof(names) / sizeof(names[0])))

static void * allocate(size_t size) {
    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, size)) abort();
    return ptr;
}

static uint32_t random_u32(uint32_t * state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void fill_x(block_q6_K * x, int nb, uint32_t seed) {
    for (int b = 0; b < nb; ++b) {
        for (int j = 0; j < QK_K/2; ++j) x[b].ql[j] = random_u32(&seed);
        for (int j = 0; j < QK_K/4; ++j) x[b].qh[j] = random_u32(&seed);
        for (int j = 0; j < QK_K/16; ++j) x[b].scales[j] = (int8_t) random_u32(&seed);
        x[b].d = ggml_fp32_to_fp16((float) (1 + random_u32(&seed) % 200) / 16384.0f);
    }
}

static void fill_y(block_q8_K * y, int nb, uint32_t seed) {
    for (int b = 0; b < nb; ++b) {
        for (int j = 0; j < QK_K; ++j) y[b].qs[j] = (int8_t) random_u32(&seed);
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int k = 0; k < 16; ++k) sum += y[b].qs[j*16 + k];
            y[b].bsums[j] = sum;
        }
        y[b].d = (float) (1 + random_u32(&seed) % 200) / 16384.0f;
    }
}


typedef struct {
    uint8_t qs[QK_K];
    int8_t scales[QK_K/16];
    float d;
} block_q6_expanded;
_Static_assert(sizeof(block_q6_expanded) == 276, "expanded block size");

static void expand(const block_q6_K * src, block_q6_expanded * dst, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        for (int j = 0; j < 2; ++j) {
            for (int g = 0; g < 4; ++g) {
                for (int t = 0; t < 32; ++t) {
                    uint8_t lo = src[b].ql[j*64+(g%2)*32+t];
                    if (g >= 2) lo >>= 4;
                    dst[b].qs[j*128+g*32+t] = (lo & 15) | (((src[b].qh[j*32+t] >> (g*2)) & 3) << 4);
                }
            }
        }
        memcpy(dst[b].scales, src[b].scales, sizeof(dst[b].scales));
        dst[b].d = ggml_fp16_to_fp32(src[b].d);
    }
}

static void exact(dot_fn * functions, int n, const void * x, const void * y) {
    static char * guarded;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t capacity = (128*sizeof(block_q6_expanded)+page-1)/page*page;
    if (!guarded) {
        guarded = mmap(NULL, capacity+page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (guarded == MAP_FAILED || mprotect(guarded+capacity,page,PROT_NONE)) abort();
    }
    block_q6_expanded * expanded = (block_q6_expanded *) (guarded+capacity-(size_t)(n/QK_K)*sizeof(block_q6_expanded));
    expand(x,expanded,n/QK_K);
    float reference;
    ggml_vec_dot_q6_K_q8_K(n, &reference, 0, x, 0, y, 0, 1);
    if (!isfinite(reference)) abort();
    for (int v = 0; v < VARIANTS; ++v) {
        float result;
        functions[v](n, &result, 0, v ? (void *) expanded : x, 0, y, 0, 1);
        if (memcmp(&reference, &result, sizeof(float))) {
            fprintf(stderr, "Mismatch variant=%s n=%d reference=%.9g result=%.9g\n",
                    names[v], n, reference, result);
            exit(3);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 6) return 1;
    const int k = atoi(argv[2]), m = atoi(argv[3]), threads = atoi(argv[4]), rounds = atoi(argv[5]);
    if (k < QK_K || k > 32768 || k % QK_K || m < 1 || m > 262144 ||
        threads < 1 || threads > 24 || rounds < 1 || rounds > 100) return 1;
    ggml_cpu_init();
    dot_fn functions[VARIANTS];
    void * handles[VARIANTS];
    for (int v = 0; v < VARIANTS; ++v) {
        char path[4096];
        if (snprintf(path, sizeof(path), "%s/%s.so", argv[1], names[v]) >= (int) sizeof(path)) return 2;
        handles[v] = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handles[v]) { fprintf(stderr, "%s\n", dlerror()); return 2; }
        *(void **) &functions[v] = dlsym(handles[v], "ggml_vec_dot_q6_K_q8_K");
        if (!functions[v]) return 2;
    }
    const long page = sysconf(_SC_PAGESIZE);
    const int blocks[] = {1, 2, 3, 4, 5, 7, 8, 11, 16, 32, 33, 64, 65, 128};
    int cases = 0;
    for (size_t i = 0; i < sizeof(blocks)/sizeof(blocks[0]); ++i) {
        int nb = blocks[i];
        size_t xb = nb*sizeof(block_q6_K), yb = nb*sizeof(block_q8_K);
        size_t xp = (xb + page-1)/page*page, yp = (yb + page-1)/page*page;
        char * xm = mmap(NULL, xp + page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        char * ym = mmap(NULL, yp + page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (xm == MAP_FAILED || ym == MAP_FAILED ||
            mprotect(xm + xp, page, PROT_NONE) || mprotect(ym + yp, page, PROT_NONE)) abort();
        block_q6_K * x = (block_q6_K *) (xm + xp-xb);
        block_q8_K * y = (block_q8_K *) (ym + yp-yb);
        for (uint32_t seed = 1; seed <= 32; ++seed) {
            fill_x(x, nb, seed);
            fill_y(y, nb, seed + 70);
            exact(functions, nb*QK_K, x, y);
            cases += VARIANTS;
        }
        munmap(xm, xp+page);
        munmap(ym, yp+page);
    }
    block_q6_K half_x;
    block_q8_K half_y;
    fill_x(&half_x, 1, 91);
    fill_y(&half_y, 1, 812);
    int half_cases = 0;
    for (uint32_t half = 0; half <= UINT16_MAX; ++half) {
        if ((half & 0x7c00) == 0x7c00) continue;
        half_x.d = (ggml_half) half;
        exact(functions, QK_K, &half_x, &half_y);
        half_cases += VARIANTS;
    }
    size_t row_bytes = (size_t) (k/QK_K)*sizeof(block_q6_K);
    char * x = allocate(row_bytes*(size_t)m);
    block_q8_K * y = allocate((size_t)(k/QK_K)*sizeof(block_q8_K));
    float * result = allocate(m*sizeof(float));
    float * reference = allocate(m*sizeof(float));
    for (int i = 0; i < m; ++i) fill_x((block_q6_K *)(x+i*row_bytes), k/QK_K, i+1);
    fill_y(y, k/QK_K, 777);
    size_t expanded_row = (size_t)(k/QK_K)*sizeof(block_q6_expanded);
    char * expanded = allocate(expanded_row*(size_t)m);
    expand((const block_q6_K *) x, (block_q6_expanded *) expanded, (size_t)(k/QK_K)*m);
    omp_set_dynamic(0);
    printf("{\"k\":%d,\"m\":%d,\"threads\":%d,\"guarded_exact_cases\":%d,\"finite_half_exact_cases\":%d,\"runs\":[",
           k,m,threads,cases,half_cases);
    for (int round = -1; round < rounds; ++round) {
        for (int order = 0; order < VARIANTS; ++order) {
            const int v = round < 0 ? order : (round % 2 ? VARIANTS-1-order : order);
            const double start = omp_get_wtime();
            #pragma omp parallel num_threads(threads)
            {
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(omp_get_thread_num(), &mask);
                if (sched_setaffinity(0,sizeof(mask),&mask)) abort();
                #pragma omp for schedule(static)
                for (int i = 0; i < m; ++i) functions[v](k,result+i,0,v ? expanded+i*expanded_row : x+i*row_bytes,0,y,0,1);
            }
            const double seconds = omp_get_wtime()-start;
            if (round == -1 && v == 0) memcpy(reference,result,m*sizeof(float));
            else if (memcmp(reference,result,m*sizeof(float))) return 3;
            if (round >= 0) printf("%s{\"round\":%d,\"variant\":\"%s\",\"seconds\":%.9f}",
                                   round || order ? "," : "", round,names[v],seconds);
        }
    }
    printf("]}\n");
    free(expanded); free(x); free(y); free(result); free(reference);
    for (int v=0; v<VARIANTS; ++v) dlclose(handles[v]);
    return 0;
}

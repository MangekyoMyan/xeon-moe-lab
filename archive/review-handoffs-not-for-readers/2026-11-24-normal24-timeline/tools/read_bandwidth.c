#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

static double read_array(const float *data, size_t count) {
    double total = 0;
    #pragma omp parallel reduction(+:total)
    {
        size_t chunks = count / 32;
        size_t first = chunks * omp_get_thread_num() / omp_get_num_threads() * 32;
        size_t last = chunks * (omp_get_thread_num() + 1) / omp_get_num_threads() * 32;
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (size_t i = first; i < last; i += 32) {
            a = _mm256_add_ps(a, _mm256_loadu_ps(data + i));
            b = _mm256_add_ps(b, _mm256_loadu_ps(data + i + 8));
            c = _mm256_add_ps(c, _mm256_loadu_ps(data + i + 16));
            d = _mm256_add_ps(d, _mm256_loadu_ps(data + i + 24));
        }
        float lanes[32];
        _mm256_storeu_ps(lanes, a);
        _mm256_storeu_ps(lanes + 8, b);
        _mm256_storeu_ps(lanes + 16, c);
        _mm256_storeu_ps(lanes + 24, d);
        for (int i = 0; i < 32; ++i) total += lanes[i];
    }
    return total;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 12;
    if (threads < 1 || threads > 256) return 2;
    const size_t bytes = (size_t)2 * 1024 * 1024 * 1024;
    const size_t count = bytes / sizeof(float);
    float *data = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    int cpus[256] = {0};
    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        cpus[thread] = sched_getcpu();
        size_t first = count / 32 * thread / omp_get_num_threads() * 32;
        size_t last = count / 32 * (thread + 1) / omp_get_num_threads() * 32;
        for (size_t i = first; i < last; ++i) data[i] = 1.0f;
    }
    if (read_array(data, count) != (double)count) return 3;
    printf("{\"bytes_per_pass\":%zu,\"threads\":%d,\"cpus\":[", bytes, threads);
    for (int i = 0; i < threads; ++i) printf("%s%d", i ? "," : "", cpus[i]);
    printf("],\"gb_s\":[");
    for (int rep = 0; rep < 9; ++rep) {
        double start = omp_get_wtime();
        double sum = read_array(data, count);
        double elapsed = omp_get_wtime() - start;
        if (sum != (double)count) return 3;
        printf("%s%.6f", rep ? "," : "", bytes / elapsed / 1e9);
    }
    printf("],\"checksum_passed\":true}\n");
    munmap(data, bytes);
    return 0;
}

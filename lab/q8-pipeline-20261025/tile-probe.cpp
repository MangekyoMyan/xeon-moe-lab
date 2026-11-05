#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "llamafile/sgemm.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

struct guarded_buffer {
    char * mapping;
    char * data;
    size_t size;
    explicit guarded_buffer(size_t bytes) {
        const size_t page = sysconf(_SC_PAGESIZE);
        const size_t aligned = (bytes + page - 1) / page * page;
        size = aligned + 2 * page;
        mapping = static_cast<char *>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        assert(mapping != MAP_FAILED);
        assert(madvise(mapping, size, MADV_NOHUGEPAGE) == 0);
        assert(mprotect(mapping, page, PROT_NONE) == 0);
        assert(mprotect(mapping + page + aligned, page, PROT_NONE) == 0);
        data = mapping + page + aligned - bytes;
    }
    ~guarded_buffer() { munmap(mapping, size); }
};
static uint32_t random_state = 0x20261025;
static uint32_t next_random() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
static void fill(block_q8_0 * data, size_t blocks) {
    for (size_t b = 0; b < blocks; ++b) {
        data[b].d = ggml_fp32_to_fp16(float(1 + next_random() % 8191) / 262144.0f);
        for (int i = 0; i < 32; ++i) data[b].qs[i] = int(next_random() % 256) - 128;
    }
}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) return 2;
    const bool validation_only = argc == 6;
    const int k = std::atoi(argv[1]), m = std::atoi(argv[2]), n = std::atoi(argv[3]), threads = std::atoi(argv[4]);
    assert(k >= 32 && k <= 16384 && k % 32 == 0 && m > 0 && m <= 16384 && n >= 2 && n <= 256);
    assert(threads > 0 && threads <= 24);
    ggml_cpu_init();
    const size_t stride = size_t(k / 32) * sizeof(block_q8_0);
    guarded_buffer weights(stride * m), inputs(stride * n), output(sizeof(float) * m * n);
    fill(reinterpret_cast<block_q8_0 *>(weights.data), size_t(m) * k / 32);
    fill(reinterpret_cast<block_q8_0 *>(inputs.data), size_t(n) * k / 32);
    std::vector<float> reference(size_t(m) * n);
    ggml_compute_params single{};
    single.nth = 1;
    ggml_cpu_sgemm_set_q8_tile(0);
    assert(llamafile_sgemm(&single, m, n, k / 32, weights.data, k / 32, inputs.data, k / 32,
                          reference.data(), m, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32));
    const int tiles[] = {0, 12, 14, 22, 23, 24, 32, 41, 42};
    omp_set_dynamic(0);
    for (int round = -1; round < (validation_only ? 1 : 4); ++round) {
        for (int order = 0; order < 9; ++order) {
            const int tile = tiles[round >= 0 && round % 2 ? 8 - order : order];
            ggml_cpu_sgemm_set_q8_tile(tile);
            std::fill_n(reinterpret_cast<float *>(output.data), size_t(m) * n, NAN);
            double started = 0, elapsed = 0;
            int iterations = 0;
            bool again = true;
            const double duration = validation_only ? 0.000001 : (round < 0 ? 0.01 : 0.35);
            #pragma omp parallel num_threads(threads) shared(started, elapsed, iterations, again)
            {
                const int tid = omp_get_thread_num();
                cpu_set_t mask;
                CPU_ZERO(&mask); CPU_SET(tid, &mask);
                assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
                ggml_compute_params params{};
                params.ith = tid;
                params.nth = threads;
                #pragma omp barrier
                #pragma omp master
                started = omp_get_wtime();
                #pragma omp barrier
                do {
                    assert(llamafile_sgemm(&params, m, n, k / 32, weights.data, k / 32, inputs.data, k / 32,
                                          output.data, m, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32));
                    #pragma omp barrier
                    #pragma omp master
                    {
                        ++iterations;
                        elapsed = omp_get_wtime() - started;
                        again = elapsed < duration;
                    }
                    #pragma omp barrier
                } while (again);
            }
            if (std::memcmp(reference.data(), output.data, reference.size() * sizeof(float))) {
                std::fprintf(stderr, "Tile mismatch code=%d K=%d M=%d N=%d threads=%d\n", tile, k, m, n, threads);
                return 3;
            }
            if (round >= 0) {
                std::printf("{\"K\":%d,\"M\":%d,\"N\":%d,\"threads\":%d,\"round\":%d,\"tile\":%d,\"iterations\":%d,\"seconds\":%.9g,\"matrix_seconds\":%.9g,\"dense_multiply_add_Gop_s\":%.9g}\n",
                            k, m, n, threads, round, tile, iterations, elapsed, elapsed / iterations,
                            2.0 * k * m * n * iterations / elapsed / 1e9);
                std::fflush(stdout);
            }
        }
    }
    ggml_cpu_sgemm_set_q8_tile(0);
}

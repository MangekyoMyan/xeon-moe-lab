#include "kernel-candidates.h"
#include "ggml-cpu.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static uint32_t rng_state = 0x4e5f6a77;
static void (*patched_q8_rows2)(int, float *, const void *, size_t, const void *);
static void (*patched_iq2)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
static int patch_outputs;

static void load_function(void * library, const char * name, void * target, size_t size) {
    void * function = dlsym(library, name);
    if (!function || size != sizeof(function)) {
        fprintf(stderr, "dlsym failed: %s\n", name);
        exit(2);
    }
    memcpy(target, &function, size);
}

static uint32_t random_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static ggml_fp16_t half_scale(int i) {
    static const ggml_fp16_t edges[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x4c01, 0x5400};
    if (i % 5 == 0) {
        return edges[(i / 5) % (sizeof(edges) / sizeof(edges[0]))];
    }
    return ggml_fp32_to_fp16(ldexpf((int) (random_u32() % 257) - 128, (int) (random_u32() % 18) - 16));
}

static void check_bits(float ref, float actual, const char * kernel, int n, int trial, int row) {
    if (float_bits(ref) != float_bits(actual)) {
        fprintf(stderr, "%s mismatch n=%d trial=%d row=%d reference=%.9g/%08x candidate=%.9g/%08x\n",
                kernel, n, trial, row, (double) ref, float_bits(ref), (double) actual, float_bits(actual));
        exit(1);
    }
}

static int check_q8(void) {
    const int sizes[] = {0, 32, 64, 96, 128, 224, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    int outputs = 0;
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        const int n = sizes[si];
        const int nb = n / QK8_0;
        for (int trial = 0; trial < 40; ++trial) {
            const size_t stride = (size_t) nb * sizeof(block_q8_0) + 2 * (trial % 9);
            const size_t row_span = stride ? stride : 2;
            unsigned char * rawx = malloc(4 * row_span + 32);
            unsigned char * rawy = malloc((size_t) nb * sizeof(block_q8_0) + 32);
            if (!rawx || !rawy) {
                exit(2);
            }
            unsigned char * x = rawx + 2 * (trial % 8);
            block_q8_0 * y = (block_q8_0 *) (rawy + 2 * ((trial + 3) % 8));
            for (int ib = 0; ib < nb; ++ib) {
                y[ib].d = half_scale(ib + trial);
                for (int k = 0; k < QK8_0; ++k) {
                    y[ib].qs[k] = (int) (random_u32() % (trial < 35 ? 255 : 256)) - (trial < 35 ? 127 : 128);
                }
                for (int r = 0; r < 4; ++r) {
                    block_q8_0 * xr = (block_q8_0 *) (x + r * row_span);
                    xr[ib].d = half_scale(ib + trial + r + 1);
                    for (int k = 0; k < QK8_0; ++k) {
                        xr[ib].qs[k] = (int) (random_u32() % 256) - 128;
                    }
                }
            }
            float ref[4];
            float s2[4] = {11, 22, 33, 44};
            float s4[6] = {11, 22, 33, 44, 55, 66};
            for (int r = 0; r < 4; ++r) {
                ggml_vec_dot_q8_0_q8_0(n, ref + r, 0, x + r * row_span, 0, y, 0, 1);
            }
            q8_rows_2(n, s2 + 1, x, row_span, y);
            q8_rows_4(n, s4 + 1, x, row_span, y);
            if (patched_q8_rows2) {
                float patched[4];
                patched_q8_rows2(n, patched, x, row_span, y);
                for (int r = 0; r < 4; ++r) {
                    check_bits(ref[r], patched[r], "q8_rows2_patch", n, trial, r);
                    ++patch_outputs;
                }
            }
            if (s2[0] != 11 || s2[3] != 44 || s4[0] != 11 || s4[5] != 66) {
                fprintf(stderr, "output canary corrupted\n");
                exit(1);
            }
            for (int r = 0; r < 4; ++r) {
                check_bits(ref[r], s4[r + 1], "q8_rows_4", n, trial, r);
                ++outputs;
                if (r < 2) {
                    check_bits(ref[r], s2[r + 1], "q8_rows_2", n, trial, r);
                    ++outputs;
                }
            }
            free(rawy);
            free(rawx);
        }
    }
    return outputs;
}

static int check_iq2(int * out_of_domain_differences) {
    const int sizes[] = {0, 256, 512, 768, 1024, 2048, 4096, 8192, 16384};
    int outputs = 0;
    *out_of_domain_differences = 0;
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        const int n = sizes[si];
        const int nb = n / QK_K;
        for (int trial = 0; trial < 120; ++trial) {
            unsigned char * rawx = malloc((size_t) nb * sizeof(block_iq2_xxs) + 32);
            unsigned char * rawy = malloc((size_t) nb * sizeof(block_q8_K) + 32);
            uint8_t * rawa = malloc((size_t) n + 32);
            block_iq2_index * packed_x = malloc((size_t) nb * sizeof(block_iq2_index) + 1);
            block_iq2_xxs * restored_x = malloc((size_t) nb * sizeof(block_iq2_xxs) + 1);
            if (!rawx || !rawy || !rawa || !packed_x || !restored_x) {
                exit(2);
            }
            block_iq2_xxs * x = (block_iq2_xxs *) (rawx + 2 * (trial % 8));
            block_q8_K * y = (block_q8_K *) (rawy + 4 * (trial % 4));
            uint8_t * ay = rawa + trial % 16;
            for (int ib = 0; ib < nb; ++ib) {
                x[ib].d = half_scale(ib + trial);
                y[ib].d = ldexpf((int) (random_u32() % 257) - 128, (int) (random_u32() % 22) - 18);
                for (int k = 0; k < QK_K / 8; ++k) {
                    x[ib].qs[k] = random_u32();
                }
                for (int k = 0; k < QK_K; ++k) {
                    y[ib].qs[k] = (int) (random_u32() % 255) - 127;
                    if (trial >= 115 && k % 8 == 0) {
                        y[ib].qs[k] = -128;
                    }
                }
                for (int k = 0; k < QK_K / 16; ++k) {
                    y[ib].bsums[k] = 0;
                    for (int j = 0; j < 16; ++j) {
                        y[ib].bsums[k] += y[ib].qs[k * 16 + j];
                    }
                }
            }
            float ref, actual, actual_abs, compact, packed;
            ggml_vec_dot_iq2_xxs_q8_K(n, &ref, 0, x, 0, y, 0, 1);
            iq2_lut_direct(n, &actual, x, y);
            iq2_activation_abs(n, y, ay);
            iq2_lut_abs(n, &actual_abs, x, y, ay);
            iq2_compact(n, &compact, x, y);
            check_bits(ref, compact, "iq2_compact", n, trial, 0);
            ++outputs;
            iq2_index_repack(n, x, packed_x);
            iq2_index_unpack(n, packed_x, restored_x);
            if (memcmp(x, restored_x, (size_t) nb * sizeof(block_iq2_xxs))) {
                fprintf(stderr, "IQ2 index repack is not reversible\n");
                exit(1);
            }
            iq2_lut_packed(n, &packed, packed_x, y);
            check_bits(ref, packed, "iq2_lut_packed", n, trial, 0);
            ++outputs;
            if (patched_iq2) {
                float patched;
                patched_iq2(n, &patched, 0, x, 0, y, 0, 1);
                check_bits(ref, patched, "iq2_lut_direct_patch", n, trial, 0);
                ++patch_outputs;
            }
            check_bits(ref, actual, "iq2_lut_direct", n, trial, 0);
            ++outputs;
            if (trial < 115) {
                check_bits(ref, actual_abs, "iq2_lut_abs", n, trial, 0);
                ++outputs;
            } else if (float_bits(ref) != float_bits(actual_abs)) {
                ++*out_of_domain_differences;
            }
            free(rawa);
            free(rawy);
            free(rawx);
            free(restored_x);
            free(packed_x);
        }
    }
    return outputs;
}

struct guard_buffer {
    void * base;
    size_t size;
    void * data;
};

static struct guard_buffer guarded(size_t bytes) {
    const long page_raw = sysconf(_SC_PAGESIZE);
    if (page_raw <= 0) {
        exit(2);
    }
    const size_t page = (size_t) page_raw;
    const size_t usable = ((bytes + page - 1) / page + 1) * page;
    struct guard_buffer buffer = {NULL, usable + 2 * page, NULL};
    buffer.base = mmap(NULL, buffer.size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buffer.base == MAP_FAILED || mprotect((char *) buffer.base + page, usable, PROT_READ | PROT_WRITE)) {
        perror("guard page setup");
        exit(2);
    }
    buffer.data = (char *) buffer.base + page + usable - bytes;
    return buffer;
}

static void unguard(struct guard_buffer buffer) {
    if (munmap(buffer.base, buffer.size)) {
        exit(2);
    }
}

static int check_guards(void) {
    const int q8_sizes[] = {0, 32, 64, 96, 256, 4096};
    const int iq2_sizes[] = {0, 256, 512, 4096};
    int calls = 0;
    for (size_t si = 0; si < sizeof(q8_sizes) / sizeof(q8_sizes[0]); ++si) {
        const int n = q8_sizes[si];
        const size_t stride = (size_t) (n / QK8_0) * sizeof(block_q8_0);
        struct guard_buffer x = guarded(stride * 4);
        struct guard_buffer y = guarded(stride);
        struct guard_buffer out2 = guarded(2 * sizeof(float));
        struct guard_buffer out4 = guarded(4 * sizeof(float));
        memset(x.data, 0x11, stride * 4);
        memset(y.data, 0x22, stride);
        q8_rows_2(n, out2.data, x.data, stride, y.data);
        q8_rows_4(n, out4.data, x.data, stride, y.data);
        calls += 2;
        if (patched_q8_rows2) {
            const void * last_two = x.data;
            patched_q8_rows2(n, out4.data, last_two, stride, y.data);
            ++calls;
        }
        unguard(out4);
        unguard(out2);
        unguard(y);
        unguard(x);
    }
    for (size_t si = 0; si < sizeof(iq2_sizes) / sizeof(iq2_sizes[0]); ++si) {
        const int n = iq2_sizes[si];
        const int nb = n / QK_K;
        struct guard_buffer x = guarded((size_t) nb * sizeof(block_iq2_xxs));
        struct guard_buffer y = guarded((size_t) nb * sizeof(block_q8_K));
        struct guard_buffer ay = guarded((size_t) n);
        struct guard_buffer packed = guarded((size_t) nb * sizeof(block_iq2_index));
        struct guard_buffer out = guarded(sizeof(float));
        memset(x.data, 0x11, (size_t) nb * sizeof(block_iq2_xxs));
        memset(y.data, 0x22, (size_t) nb * sizeof(block_q8_K));
        iq2_lut_direct(n, out.data, x.data, y.data);
        iq2_compact(n, out.data, x.data, y.data);
        iq2_activation_abs(n, y.data, ay.data);
        iq2_lut_abs(n, out.data, x.data, y.data, ay.data);
        iq2_index_repack(n, x.data, packed.data);
        iq2_lut_packed(n, out.data, packed.data, y.data);
        iq2_index_unpack(n, packed.data, x.data);
        calls += 7;
        if (patched_iq2) {
            patched_iq2(n, out.data, 0, x.data, 0, y.data, 0, 1);
            ++calls;
        }
        unguard(out);
        unguard(packed);
        unguard(ay);
        unguard(y);
        unguard(x);
    }
    return calls;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    iq2_lut_init();
    void * q8_library = NULL;
    void * iq2_library = NULL;
    if (argc == 3) {
        q8_library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        iq2_library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        if (!q8_library || !iq2_library) {
            fprintf(stderr, "dlopen failed: %s\n", dlerror());
            return 2;
        }
        load_function(q8_library, "ggml_vec_dot_q8_0_q8_0_rows4", &patched_q8_rows2, sizeof(patched_q8_rows2));
        load_function(iq2_library, "ggml_vec_dot_iq2_xxs_q8_K", &patched_iq2, sizeof(patched_iq2));
        void (*initialize_iq2)(void);
        load_function(iq2_library, "ggml_iq2_xxs_init_avx", &initialize_iq2, sizeof(initialize_iq2));
        initialize_iq2();
    } else if (argc != 1) {
        fprintf(stderr, "usage: kernel-probe [q8-patch.so iq2-patch.so]\n");
        return 2;
    }
    const int lut_bytes_checked = iq2_lut_check();
    if (lut_bytes_checked != 256 * 128 * 8) {
        fprintf(stderr, "LUT check failed\n");
        return 1;
    }
    const int q8_outputs = check_q8();
    int out_of_domain_differences = 0;
    const int iq2_outputs = check_iq2(&out_of_domain_differences);
    const int guard_calls = check_guards();
    if (out_of_domain_differences == 0) {
        fprintf(stderr, "negative test did not expose the -128 domain limit\n");
        return 1;
    }
    printf("{\"status\":\"pass\",\"compiled_isa\":\"ivybridge-avx-no-avx2-no-fma\","
           "\"lut_bytes_exhaustively_checked\":%d,\"q8_bit_identical_outputs\":%d,"
           "\"iq2_bit_identical_outputs\":%d,\"actual_patch_bit_identical_outputs\":%d,\"guard_page_calls\":%d,\"iq2_abs_out_of_domain_differences\":%d}\n",
           lut_bytes_checked, q8_outputs, iq2_outputs, patch_outputs, guard_calls, out_of_domain_differences);
    if (q8_library) {
        dlclose(q8_library);
        dlclose(iq2_library);
    }
    return 0;
}

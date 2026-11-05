#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static ggml_vec_dot_t original;

static __m128i dot16(__m128i x, __m128i y) {
    return _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(x, x), _mm_sign_epi8(y, x)), _mm_set1_epi16(1));
}

static void candidate(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    if (n % 64) {
        original(n, s, bs, vx, bx, vy, by, nrc);
        return;
    }
    const auto * x = static_cast<const block_q8_0 *>(vx);
    const auto * y = static_cast<const block_q8_0 *>(vy);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 32; i += 2) {
        const __m128i a = _mm_add_epi32(dot16(_mm_loadu_si128((const __m128i *) x[i].qs), _mm_loadu_si128((const __m128i *) y[i].qs)),
                                      dot16(_mm_loadu_si128((const __m128i *) (x[i].qs + 16)), _mm_loadu_si128((const __m128i *) (y[i].qs + 16))));
        const __m128i b = _mm_add_epi32(dot16(_mm_loadu_si128((const __m128i *) x[i+1].qs), _mm_loadu_si128((const __m128i *) y[i+1].qs)),
                                      dot16(_mm_loadu_si128((const __m128i *) (x[i+1].qs + 16)), _mm_loadu_si128((const __m128i *) (y[i+1].qs + 16))));
        const __m256 q = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(a), b, 1));
        const __m128i halves = _mm_set_epi32(0, 0, (uint32_t(y[i+1].d) << 16) | y[i].d, (uint32_t(x[i+1].d) << 16) | x[i].d);
        const __m128 scales = _mm_cvtph_ps(halves);
        const __m128 products = _mm_mul_ps(scales, _mm_movehl_ps(scales, scales));
        const __m256 d = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_shuffle_ps(products, products, 0)), _mm_shuffle_ps(products, products, 0x55), 1);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(q, d));
    }
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    *s = _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
}

int main() {
    ggml_cpu_init();
    original = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale(0.0001f, 0.1f);
    const int n = 4096;
    const int rows = 32768;
    std::vector<block_q8_0> x(size_t(rows) * n / 32), y(n / 32);
    for (auto * data : { &x, &y }) {
        for (auto & block : *data) {
            block.d = ggml_fp32_to_fp16(scale(rng));
            for (auto & q : block.qs) q = int(rng() % 255) - 127;
        }
    }
    float max_error = 0;
    for (int row = 0; row < 1024; row++) {
        float a, b;
        original(n, &a, 0, x.data() + row*n/32, 0, y.data(), 0, 1);
        candidate(n, &b, 0, x.data() + row*n/32, 0, y.data(), 0, 1);
        max_error = std::max(max_error, std::abs(a-b));
    }
    printf("maximum absolute difference: %.9g\n", max_error);
    if (max_error != 0) return 1;
    volatile float sink = 0;
    for (int active_rows : { 1, rows }) {
        for (int repeat = 0; repeat < 6; repeat++) {
            auto fn = repeat % 2 ? candidate : original;
            const auto start = std::chrono::steady_clock::now();
            for (int row = 0; row < rows; row++) {
                float result;
                fn(n, &result, 0, x.data() + (row % active_rows)*n/32, 0, y.data(), 0, 1);
                sink = result;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("rows=%d implementation=%s seconds=%.6f\n", active_rows, repeat % 2 ? "f16c" : "original", seconds);
        }
    }
    return sink == 123456789.0f;
}

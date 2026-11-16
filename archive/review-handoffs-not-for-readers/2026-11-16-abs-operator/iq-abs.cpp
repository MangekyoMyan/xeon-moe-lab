// Adapted from the reviewed Pro input-abs fixture; compact weight bytes are unchanged.
#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu/simd-mappings.h"
#include "iq-abs.h"
#include <atomic>
#include <cstring>
#include <type_traits>
#include <immintrin.h>
#include "iq-small-kernels.inc"

static std::atomic<int> mode{0}, diagnostic{0};
static std::atomic<uint64_t> totals[3]{};
void xeon_iq_abs_set_mode(int value) { GGML_ASSERT(value >= 0 && value <= 2); mode.store(value); }
int xeon_iq_abs_get_mode() { return mode.load(std::memory_order_relaxed); }
void xeon_iq_abs_set_diagnostics(int value) { diagnostic.store(value != 0); }
int xeon_iq_abs_diagnostics() { return diagnostic.load(std::memory_order_relaxed); }
void xeon_iq_abs_counts(uint64_t out[3]) { for (int i=0; i<3; ++i) out[i]=totals[i].load(); }
void xeon_iq_abs_report(uint64_t blocks, uint64_t rows, uint64_t unsafe) {
    totals[0].fetch_add(blocks, std::memory_order_relaxed);
    totals[1].fetch_add(rows, std::memory_order_relaxed);
    totals[2].fetch_add(unsafe, std::memory_order_relaxed);
}
int xeon_iq_abs_prepare(int n, const void * input, int first, int end, void * output) {
    GGML_ASSERT(n > 0 && n % 256 == 0 && first >= 0 && first <= end && end <= n/256);
    const auto * x = static_cast<const block_q8_K *>(input);
    auto * dst = static_cast<uint8_t *>(output);
    const __m128i minimum = _mm_set1_epi8(-128);
    bool safe = true;
    for (int b=first; b<end; ++b) {
        uint32_t bits; std::memcpy(&bits, &x[b].d, 4);
        if ((bits & 0x7f800000u) == 0x7f800000u) safe = false;
        for (int h=0; h<16; ++h) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16*h));
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(v, minimum))) safe = false;
            _mm_store_si128(reinterpret_cast<__m128i *>(dst+256*b+16*h), _mm_abs_epi8(v));
        }
    }
    return safe;
}
template<bool two> static float dot(int n, const iq_compact_block * w, const block_q8_K * x, const uint8_t * prepared) {
    const __m128i lut = two ? _mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0) :
                             _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (int b=0; b<n/256; ++b) {
        __m128i low = _mm_setzero_si128(), high = low;
        for (int g=0; g<8; ++g) {
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0 = _mm_shuffle_epi8(lut, _mm_and_si128(bits, mask));
            const __m128i q1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(bits,4), mask));
            const __m128i y0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g));
            const __m128i y1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16));
            const __m128i a0 = _mm_load_si128(reinterpret_cast<const __m128i *>(prepared+256*b+32*g));
            const __m128i a1 = _mm_load_si128(reinterpret_cast<const __m128i *>(prepared+256*b+32*g+16));
            low = _mm_add_epi32(low, _mm_madd_epi16(_mm_maddubs_epi16(a0,_mm_sign_epi8(q0,y0)), _mm_set1_epi16(iq_scale<two>(w[b].scales,g,0))));
            high = _mm_add_epi32(high, _mm_madd_epi16(_mm_maddubs_epi16(a1,_mm_sign_epi8(q1,y1)), _mm_set1_epi16(iq_scale<two>(w[b].scales,g,1))));
        }
        const __m256i ints = _mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)),acc);
    }
    return (two ? .125f : 1.f)*iq_hsum(acc);
}
void xeon_iq2_abs_dot(int n,float *out,const void *w,const void *x,const void *p) {
    *out=dot<true>(n,static_cast<const iq_compact_block *>(w),static_cast<const block_q8_K *>(x),static_cast<const uint8_t *>(p));
}
void xeon_iq3s_abs_dot(int n,float *out,const void *w,const void *x,const void *p) {
    *out=dot<false>(n,static_cast<const iq_compact_block *>(w),static_cast<const block_q8_K *>(x),static_cast<const uint8_t *>(p));
}

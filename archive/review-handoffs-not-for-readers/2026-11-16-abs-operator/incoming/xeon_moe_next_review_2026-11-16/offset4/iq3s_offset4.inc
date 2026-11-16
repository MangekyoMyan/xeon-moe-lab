// Experimental IQ3_S recoding. This is not the current compact weight schema.
struct alignas(16) offset4_iq3s_block {
    uint8_t qs[128];
    float d;
    uint8_t scales[8];
};
struct alignas(16) offset4_rhs_block { int16_t pairs[16][8]; };
static_assert(sizeof(offset4_iq3s_block) == 144, "weight payload");
static_assert(sizeof(offset4_rhs_block) == 256, "input sidecar");
static constexpr const char * offset4_schema = "IQ3S_OFFSET4_V1_144";

static void offset4_pack(const iq_compact_block & source, offset4_iq3s_block & dest) {
    static constexpr uint8_t map[16] = {8,9,10,11,12,13,14,15,7,6,5,4,3,2,1,0};
    for (int i = 0; i < 128; ++i) {
        dest.qs[i] = map[source.qs[i] & 15] | (map[source.qs[i] >> 4] << 4);
    }
    dest.d = source.d;
    std::memcpy(dest.scales, source.scales, 8);
}

static bool offset4_prepare(int n, const block_q8_K * input, offset4_rhs_block * dest) {
    const __m128i bias = _mm_set1_epi8(15), minimum = _mm_set1_epi8(-128);
    bool safe = true;
    for (int b = 0; b < n/256; ++b) {
        uint32_t bits;
        std::memcpy(&bits, &input[b].d, 4);
        if ((bits & 0x7f800000u) == 0x7f800000u) safe = false;
        for (int h = 0; h < 16; ++h) {
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input[b].qs + 16*h));
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(q, minimum))) safe = false;
            _mm_store_si128(reinterpret_cast<__m128i *>(dest[b].pairs[h]), _mm_maddubs_epi16(bias, q));
        }
    }
    return safe;
}

// Call only after offset4_prepare returned true. Per-row integer/FP order is unchanged.
__attribute__((noinline)) static float offset4_fast_dot(int n, const offset4_iq3s_block * w,
        const block_q8_K * x, const offset4_rhs_block * correction) {
    const __m128i mask = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n/256; ++b) {
        __m128i low = _mm_setzero_si128(), high = low;
        for (int g = 0; g < 8; ++g) {
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs + 16*g));
            const __m128i q0 = _mm_and_si128(bits, mask);
            const __m128i q1 = _mm_and_si128(_mm_srli_epi16(bits, 4), mask);
            const __m128i y0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g));
            const __m128i y1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g + 16));
            const __m128i c0 = _mm_load_si128(reinterpret_cast<const __m128i *>(correction[b].pairs[2*g]));
            const __m128i c1 = _mm_load_si128(reinterpret_cast<const __m128i *>(correction[b].pairs[2*g+1]));
            const __m128i p0 = _mm_maddubs_epi16(q0, y0);
            const __m128i p1 = _mm_maddubs_epi16(q1, y1);
            const __m128i signed0 = _mm_sub_epi16(_mm_add_epi16(p0, p0), c0);
            const __m128i signed1 = _mm_sub_epi16(_mm_add_epi16(p1, p1), c1);
            const __m128i scale = _mm_set1_epi16(iq_scale<false>(w[b].scales, g, 0));
            low = _mm_add_epi32(low, _mm_madd_epi16(signed0, scale));
            high = _mm_add_epi32(high, _mm_madd_epi16(signed1, scale));
        }
        const __m256i integers = _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
        const __m256 product = _mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d), _mm256_cvtepi32_ps(integers));
        acc = _mm256_add_ps(product, acc);
    }
    return iq_hsum(acc);
}

// The old compact reader cannot read offset4 weights. This reader preserves its arithmetic.
__attribute__((noinline)) static float offset4_compat_dot(int n, const offset4_iq3s_block * w,
        const block_q8_K * x) {
    const __m128i lut = _mm_setr_epi8(-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15);
    const __m128i mask = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n/256; ++b) {
        __m128i low = _mm_setzero_si128(), high = low;
        for (int g = 0; g < 8; ++g) {
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs + 16*g));
            const __m128i q0 = _mm_shuffle_epi8(lut, _mm_and_si128(bits, mask));
            const __m128i q1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(bits, 4), mask));
            const __m128i y0 = _mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g)), q0);
            const __m128i y1 = _mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g + 16)), q1);
            const __m128i scale = _mm_set1_epi16(iq_scale<false>(w[b].scales, g, 0));
            low = _mm_add_epi32(low, _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0), y0), scale));
            high = _mm_add_epi32(high, _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1), y1), scale));
        }
        const __m256i integers = _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
        const __m256 product = _mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d), _mm256_cvtepi32_ps(integers));
        acc = _mm256_add_ps(product, acc);
    }
    return iq_hsum(acc);
}

// A code byte holds one sign bit for each half of a 32-value group.
struct folded_group {
    uint8_t signs[16];
    uint8_t magnitude[2][16];
};

struct folded_block {
    folded_group group[8];
    float d;
    uint8_t has_min;
    uint8_t padding[11];
};
static_assert(sizeof(folded_block) == XEON_IQ_INPUT_BLOCK_BYTES, "input stride");

static inline void fold_q8_block(const block_q8_K & q, folded_block & dst) {
    dst.d = q.d;
    __m128i any_magnitude = _mm_setzero_si128();
    for (int g = 0; g < 8; ++g) {
        auto & group = dst.group[g];
        const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q.qs + 32*g));
        const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q.qs + 32*g + 16));
        const __m128i sign_lo = _mm_and_si128(_mm_srli_epi16(lo, 4), _mm_set1_epi8(0x08));
        const __m128i sign_hi = _mm_and_si128(hi, _mm_set1_epi8(-128));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(group.signs), _mm_or_si128(sign_lo, sign_hi));
        const __m128i a0 = _mm_abs_epi8(lo);
        const __m128i a1 = _mm_abs_epi8(hi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(group.magnitude[0]), a0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(group.magnitude[1]), a1);
        any_magnitude = _mm_or_si128(any_magnitude, _mm_or_si128(a0, a1));
    }
    dst.has_min = _mm_movemask_epi8(any_magnitude) != 0;
}

template<ggml_type type>
static void folded_dot(int n, float * result, size_t, const void * weights, size_t, const void * input, size_t, int nrc) {
    GGML_ASSERT(nrc == 1 && n%QK_K == 0);
    constexpr bool xxs = type == GGML_TYPE_IQ3_XXS;
    constexpr bool two = type == GGML_TYPE_IQ2_S;
    using weight_block = std::conditional_t<xxs, compact_block, iq_compact_block>;
    const auto * w = static_cast<const weight_block *>(weights);
    const auto * x = static_cast<const folded_block *>(input);
    const __m128i lut = xxs ? _mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62)
                       : two ? _mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0)
                             : _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n/QK_K; ++b) {
        __m128i lo = _mm_setzero_si128();
        __m128i hi = _mm_setzero_si128();
        for (int g = 0; g < 8; ++g) {
            const folded_group & group = x[b].group[g];
            const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs + 16*g));
            const __m128i sign = _mm_loadu_si128(reinterpret_cast<const __m128i *>(group.signs));
            const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(group.magnitude[0]));
            const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(group.magnitude[1]));
            __m128i codes = _mm_xor_si128(qs, sign);
            if (x[b].has_min) {
                // psignb(-128, +/-weight) is -128 for either sign of the original weight.
                const __m128i force_lo = _mm_and_si128(_mm_srli_epi16(a0, 4), _mm_set1_epi8(8));
                const __m128i force_hi = _mm_and_si128(a1, _mm_set1_epi8(-128));
                codes = _mm_or_si128(codes, _mm_or_si128(force_lo, force_hi));
            }
            const __m128i q0 = _mm_shuffle_epi8(lut, _mm_and_si128(codes, mask));
            const __m128i q1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(codes, 4), mask));
            int scale0, scale1;
            if constexpr (xxs) {
                scale0 = scale1 = w[b].scale[g];
            } else {
                scale0 = iq_scale<two>(w[b].scales, g, 0);
                scale1 = iq_scale<two>(w[b].scales, g, 1);
            }
            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_maddubs_epi16(a0, q0), _mm_set1_epi16(scale0)));
            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_maddubs_epi16(a1, q1), _mm_set1_epi16(scale1)));
        }
        const __m256i ints = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d), _mm256_cvtepi32_ps(ints)));
    }
    *result = (xxs ? 0.25f : two ? 0.125f : 1.f)*iq_hsum(acc);
}

ggml_vec_dot_t xeon_iq_input_dot(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS: return folded_dot<GGML_TYPE_IQ3_XXS>;
        case GGML_TYPE_IQ2_S:   return folded_dot<GGML_TYPE_IQ2_S>;
        case GGML_TYPE_IQ3_S:   return folded_dot<GGML_TYPE_IQ3_S>;
        default:                GGML_ABORT("unsupported compact weight type");
    }
}

void xeon_iq_input_from_q8(const void * input, void * output, size_t count) {
    const auto * src = static_cast<const block_q8_K *>(input);
    auto * dst = static_cast<folded_block *>(output);
    for (size_t b=0;b<count;++b) fold_q8_block(src[b], dst[b]);
}

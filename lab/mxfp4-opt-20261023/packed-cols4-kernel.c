void ggml_vec_dot_mxfp4_q8_0_packed_cols4(int n, float * s, const void * vx, int row, const void * const vy[4]) {
    assert(n >= 0 && n % QK_MXFP4 == 0 && row >= 0 && row < 4);
    const int nb = n / QK_MXFP4;
    const uint8_t * packed = vx;
    const block_q8_0 * y[4] = { vy[0], vy[1], vy[2], vy[3] };
    __m256 acc[4];
    for (int c = 0; c < 4; ++c) acc[c] = _mm256_setzero_ps();
    const __m128i values = _mm_loadu_si128((const __m128i *) kvalues_fp4);
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i ones = _mm_set1_epi16(1);
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i bits0 = _mm_loadu_si128((const __m128i *) (packed + (size_t) ib * 68 + 4 + row * 16));
        const __m128i bits1 = _mm_loadu_si128((const __m128i *) (packed + (size_t) (ib + 1) * 68 + 4 + row * 16));
        const __m128i q00 = _mm_shuffle_epi8(values, _mm_and_si128(bits0, mask));
        const __m128i q01 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits0, 4), mask));
        const __m128i q10 = _mm_shuffle_epi8(values, _mm_and_si128(bits1, mask));
        const __m128i q11 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits1, 4), mask));
        const __m128i a00 = _mm_sign_epi8(q00, q00);
        const __m128i a01 = _mm_sign_epi8(q01, q01);
        const __m128i a10 = _mm_sign_epi8(q10, q10);
        const __m128i a11 = _mm_sign_epi8(q11, q11);
        const float xd0 = GGML_CPU_E8M0_TO_FP32_HALF(packed[(size_t) ib * 68 + row]);
        const float xd1 = GGML_CPU_E8M0_TO_FP32_HALF(packed[(size_t) (ib + 1) * 68 + row]);
        for (int c = 0; c < 4; ++c) {
            const __m128i y00 = _mm_loadu_si128((const __m128i *) y[c][ib].qs);
            const __m128i y01 = _mm_loadu_si128((const __m128i *) (y[c][ib].qs + 16));
            const __m128i y10 = _mm_loadu_si128((const __m128i *) y[c][ib + 1].qs);
            const __m128i y11 = _mm_loadu_si128((const __m128i *) (y[c][ib + 1].qs + 16));
            const __m128i p16_0 = _mm_add_epi16(_mm_maddubs_epi16(a00, _mm_sign_epi8(y00, q00)), _mm_maddubs_epi16(a01, _mm_sign_epi8(y01, q01)));
            const __m128i p16_1 = _mm_add_epi16(_mm_maddubs_epi16(a10, _mm_sign_epi8(y10, q10)), _mm_maddubs_epi16(a11, _mm_sign_epi8(y11, q11)));
            const __m128i p0 = _mm_madd_epi16(p16_0, ones);
            const __m128i p1 = _mm_madd_epi16(p16_1, ones);
            const __m256 p = _mm256_cvtepi32_ps(MM256_SET_M128I(p1, p0));
            const __m256 d = _mm256_set_m128(_mm_set1_ps(xd1 * GGML_CPU_FP16_TO_FP32(y[c][ib + 1].d)), _mm_set1_ps(xd0 * GGML_CPU_FP16_TO_FP32(y[c][ib].d)));
            acc[c] = _mm256_add_ps(_mm256_mul_ps(d, p), acc[c]);
        }
    }
    for (int c = 0; c < 4; ++c) {
        float sum = hsum_float_8(acc[c]);
        if (ib < nb) {
            const uint8_t * qs = packed + (size_t) ib * 68 + 4 + row * 16;
            int a = 0, b = 0;
            for (int j = 0; j < 16; ++j) {
                a += y[c][ib].qs[j] * kvalues_fp4[qs[j] & 15];
                b += y[c][ib].qs[j + 16] * kvalues_fp4[qs[j] >> 4];
            }
            sum += (GGML_CPU_FP16_TO_FP32(y[c][ib].d) * GGML_CPU_E8M0_TO_FP32_HALF(packed[(size_t) ib * 68 + row])) * (a + b);
        }
        s[c] = sum;
    }
}

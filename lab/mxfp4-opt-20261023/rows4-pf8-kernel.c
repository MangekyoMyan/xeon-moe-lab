
void ggml_vec_dot_mxfp4_q8_0_grouped(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    if (nrc == 1) { ggml_vec_dot_mxfp4_q8_0(n, s, bs, vx, bx, vy, by, nrc); return; }
    assert(nrc == 4 && n >= 0 && n % 32 == 0);
    const int nb = n / 32;
    const block_mxfp4 * x[4];
    const block_q8_0 * y = vy;
    __m256 acc[4];
    for (int r = 0; r < 4; ++r) {
        x[r] = (const block_mxfp4 *) ((const char *) vx + r * bx);
        acc[r] = _mm256_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i values = _mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0);
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i y00 = _mm_loadu_si128((const __m128i *) y[ib].qs);
        const __m128i y01 = _mm_loadu_si128((const __m128i *) (y[ib].qs + 16));
        const __m128i y10 = _mm_loadu_si128((const __m128i *) y[ib + 1].qs);
        const __m128i y11 = _mm_loadu_si128((const __m128i *) (y[ib + 1].qs + 16));
        const float yd0 = GGML_CPU_FP16_TO_FP32(y[ib].d);
        const float yd1 = GGML_CPU_FP16_TO_FP32(y[ib + 1].d);
        const __m128i offset = _mm_set1_epi8(12);
        const __m128i corr0 = _mm_add_epi16(_mm_maddubs_epi16(offset, y00), _mm_maddubs_epi16(offset, y01));
        const __m128i corr1 = _mm_add_epi16(_mm_maddubs_epi16(offset, y10), _mm_maddubs_epi16(offset, y11));
        for (int r = 0; r < 4; ++r) {
            if (ib + 8 < nb) {
                _mm_prefetch((const char *) &x[r][ib + 8], _MM_HINT_T0);
            }
            const __m128i bits0 = _mm_loadu_si128((const __m128i *) x[r][ib].qs);
            const __m128i bits1 = _mm_loadu_si128((const __m128i *) x[r][ib + 1].qs);
            const __m128i q00 = _mm_shuffle_epi8(values, _mm_and_si128(bits0, mask));
            const __m128i q01 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits0, 4), mask));
            const __m128i q10 = _mm_shuffle_epi8(values, _mm_and_si128(bits1, mask));
            const __m128i q11 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits1, 4), mask));
            const __m128i p0 = _mm_madd_epi16(_mm_sub_epi16(_mm_add_epi16(_mm_maddubs_epi16(q00, y00), _mm_maddubs_epi16(q01, y01)), corr0), ones);
            const __m128i p1 = _mm_madd_epi16(_mm_sub_epi16(_mm_add_epi16(_mm_maddubs_epi16(q10, y10), _mm_maddubs_epi16(q11, y11)), corr1), ones);
            const __m256 p = _mm256_cvtepi32_ps(MM256_SET_M128I(p1, p0));
            const __m256 d = _mm256_set_m128(_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[r][ib + 1].e) * yd1), _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[r][ib].e) * yd0));
            acc[r] = _mm256_add_ps(_mm256_mul_ps(d, p), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        float sum = hsum_float_8(acc[r]);
        if (ib < nb) {
            int a = 0, b = 0;
            for (int j = 0; j < 16; ++j) {
                a += y[ib].qs[j] * kvalues_fp4[x[r][ib].qs[j] & 15];
                b += y[ib].qs[j + 16] * kvalues_fp4[x[r][ib].qs[j] >> 4];
            }
            sum += (GGML_CPU_FP16_TO_FP32(y[ib].d) * GGML_CPU_E8M0_TO_FP32_HALF(x[r][ib].e)) * (a + b);
        }
        s[r] = sum;
    }
}

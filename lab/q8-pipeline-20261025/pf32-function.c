void ivb_q8_pf32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        if (ib + 32 < nb) _mm_prefetch((const char *) &x[ib + 32], _MM_HINT_T0);
        const __m128i x0_0_0 = _mm_loadu_si128((const __m128i *) x[ib + 0].qs + 0);
        const __m128i x0_0_1 = _mm_loadu_si128((const __m128i *) x[ib + 0].qs + 1);
        const __m128i x0_1_0 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs + 0);
        const __m128i x0_1_1 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs + 1);
        const __m128i y0_0_0 = _mm_loadu_si128((const __m128i *) y[ib + 0].qs + 0);
        const __m128i y0_0_1 = _mm_loadu_si128((const __m128i *) y[ib + 0].qs + 1);
        const __m128i y0_1_0 = _mm_loadu_si128((const __m128i *) y[ib + 1].qs + 0);
        const __m128i y0_1_1 = _mm_loadu_si128((const __m128i *) y[ib + 1].qs + 1);
        const __m256 p0 = mul_sum_i8_quad_float(x0_0_0, x0_0_1, x0_1_0, x0_1_1, y0_0_0, y0_0_1, y0_1_0, y0_1_1);
        const __m256 d0 = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(d0, p0), accum);
    }
    sumf = hsum_float_8(accum);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

static inline float prepared_dot(int n, const compact_block * w, const block_q8_K * x, const rhs_correction * correction) {
    const __m128i lut = _mm_setr_epi8(66,74,82,90,98,106,114,124,58,50,42,34,26,18,10,0);
    const __m128i mask = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n/256; ++b) {
        __m128i low = _mm_setzero_si128(), high = low;
        for (int g = 0; g < 8; ++g) {
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs + 16*g));
            const __m128i q0 = _mm_shuffle_epi8(lut, _mm_and_si128(bits, mask));
            const __m128i q1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(bits, 4), mask));
            const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g));
            const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs + 32*g + 16));
            const __m128i c0 = _mm_load_si128(reinterpret_cast<const __m128i *>(correction[b].pairs[2*g]));
            const __m128i c1 = _mm_load_si128(reinterpret_cast<const __m128i *>(correction[b].pairs[2*g+1]));
            const __m128i p0 = _mm_sub_epi16(_mm_maddubs_epi16(q0, x0), c0);
            const __m128i p1 = _mm_sub_epi16(_mm_maddubs_epi16(q1, x1), c1);
            const __m128i scale = _mm_set1_epi16(w[b].scale[g]);
            low = _mm_add_epi32(low, _mm_madd_epi16(p0, scale));
            high = _mm_add_epi32(high, _mm_madd_epi16(p1, scale));
        }
        const __m256i ints = _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d), _mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}


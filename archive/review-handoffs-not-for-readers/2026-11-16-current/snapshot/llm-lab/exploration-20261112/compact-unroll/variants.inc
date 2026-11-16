__attribute__((noinline)) static float compact_dot_u1(int n,const compact_block *w,const block_q8_K *x) {
    const __m128i lut=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const __m128i mask=_mm_set1_epi8(15);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 1
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            const __m128i scale=_mm_set1_epi16(w[b].scale[g]);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),scale));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),scale));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}


__attribute__((noinline)) static float compact_dot_u2(int n,const compact_block *w,const block_q8_K *x) {
    const __m128i lut=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const __m128i mask=_mm_set1_epi8(15);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 2
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            const __m128i scale=_mm_set1_epi16(w[b].scale[g]);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),scale));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),scale));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}


__attribute__((noinline)) static float compact_dot_u4(int n,const compact_block *w,const block_q8_K *x) {
    const __m128i lut=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const __m128i mask=_mm_set1_epi8(15);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 4
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            const __m128i scale=_mm_set1_epi16(w[b].scale[g]);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),scale));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),scale));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}


__attribute__((noinline)) static float compact_dot_u8(int n,const compact_block *w,const block_q8_K *x) {
    const __m128i lut=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const __m128i mask=_mm_set1_epi8(15);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 8
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            const __m128i scale=_mm_set1_epi16(w[b].scale[g]);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),scale));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),scale));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}


template<bool two> __attribute__((noinline)) static float iq_compact_dot_u1(int n,const iq_compact_block *w,const block_q8_K *x) {
    const __m128i lut=two?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                         _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);__m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 1
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,0))));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,1))));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)),acc);
    }
    return (two?.125f:1.f)*iq_hsum(acc);
}

template<bool two> __attribute__((noinline)) static float iq_compact_dot_u2(int n,const iq_compact_block *w,const block_q8_K *x) {
    const __m128i lut=two?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                         _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);__m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 2
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,0))));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,1))));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)),acc);
    }
    return (two?.125f:1.f)*iq_hsum(acc);
}

template<bool two> __attribute__((noinline)) static float iq_compact_dot_u4(int n,const iq_compact_block *w,const block_q8_K *x) {
    const __m128i lut=two?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                         _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);__m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 4
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,0))));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,1))));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)),acc);
    }
    return (two?.125f:1.f)*iq_hsum(acc);
}

template<bool two> __attribute__((noinline)) static float iq_compact_dot_u8(int n,const iq_compact_block *w,const block_q8_K *x) {
    const __m128i lut=two?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                         _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);__m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 8
        for(int g=0;g<8;++g) {
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            const __m128i y0=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g)),q0);
            const __m128i y1=_mm_sign_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16)),q1);
            low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,0))));
            high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),_mm_set1_epi16(iq_scale<two>(w[b].scales,g,1))));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(w[b].d*x[b].d),_mm256_cvtepi32_ps(ints)),acc);
    }
    return (two?.125f:1.f)*iq_hsum(acc);
}

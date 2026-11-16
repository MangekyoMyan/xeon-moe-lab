// Two output rows share Q8 activation loads; each row keeps the original reduction order.
template<int Kind> using pair_block = typename std::conditional<Kind==0,compact_block,iq_compact_block>::type;
template<int Kind> static inline int pair_scale(const pair_block<Kind> &w,int g,int h) {
    if constexpr(Kind==0)return w.scale[g];
    else return iq_scale<Kind==1>(w.scales,g,h);
}
template<int Kind> static inline void pair_accumulate(const pair_block<Kind> &w,int g,
        __m128i x0,__m128i x1,__m128i lut,__m128i mask,__m128i &low,__m128i &high) {
    const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w.qs+16*g));
    const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
    const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
    const __m128i y0=_mm_sign_epi8(x0,q0),y1=_mm_sign_epi8(x1,q1);
    low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),y0),_mm_set1_epi16(pair_scale<Kind>(w,g,0))));
    high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),y1),_mm_set1_epi16(pair_scale<Kind>(w,g,1))));
}
template<int Kind> __attribute__((noinline)) static void paired_dot(int n,float out[2],
        const pair_block<Kind> *w0,const pair_block<Kind> *w1,const block_q8_K *x) {
    const __m128i lut=Kind==0?_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62):
        Kind==1?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);
    __m256 acc0=_mm256_setzero_ps(),acc1=acc0;
    for(int b=0;b<n/256;++b) {
        __m128i lo0=_mm_setzero_si128(),hi0=lo0,lo1=lo0,hi1=lo0;
        #pragma GCC unroll 8
        for(int g=0;g<8;++g) {
            const __m128i x0=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g));
            const __m128i x1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16));
            pair_accumulate<Kind>(w0[b],g,x0,x1,lut,mask,lo0,hi0);
            pair_accumulate<Kind>(w1[b],g,x0,x1,lut,mask,lo1,hi1);
        }
        const __m256i i0=_mm256_insertf128_si256(_mm256_castsi128_si256(lo0),hi0,1);
        const __m256i i1=_mm256_insertf128_si256(_mm256_castsi128_si256(lo1),hi1,1);
        const __m256 f0=_mm256_mul_ps(_mm256_set1_ps(w0[b].d*x[b].d),_mm256_cvtepi32_ps(i0));
        const __m256 f1=_mm256_mul_ps(_mm256_set1_ps(w1[b].d*x[b].d),_mm256_cvtepi32_ps(i1));
        if constexpr(Kind==0) {acc0=_mm256_add_ps(acc0,f0);acc1=_mm256_add_ps(acc1,f1);}
        else {acc0=_mm256_add_ps(f0,acc0);acc1=_mm256_add_ps(f1,acc1);}
    }
    if constexpr(Kind==0) {out[0]=reduce(acc0);out[1]=reduce(acc1);}
    else {out[0]=(Kind==1?.125f:1.f)*iq_hsum(acc0);out[1]=(Kind==1?.125f:1.f)*iq_hsum(acc1);}
}

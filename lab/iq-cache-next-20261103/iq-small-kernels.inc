struct alignas(16) iq_compact_block { uint8_t qs[128]; float d; uint8_t scales[8]; };
static_assert(sizeof(iq_compact_block)==144,"compact size");
static const auto iq_signs=[] {
    std::array<uint64_t,256> table{};
    for(int s=0;s<256;++s)for(int j=0;j<8;++j)table[s]|=uint64_t((s&(1<<j))?255:1)<<(8*j);
    return table;
}();
static inline float iq_hsum(__m256 x) {
    __m128 sum=_mm_add_ps(_mm256_extractf128_ps(x,1),_mm256_castps256_ps128(x));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}
template<bool two> using iq_source = typename std::conditional<two,block_iq2_s,block_iq3_s>::type;
template<bool two> static inline __m128i iq_grid_half(const iq_source<two> &w,int g,int h) {
    if constexpr(two) {
        const int j=4*g+2*h;
        return _mm_set_epi64x(iq2s_grid[w.qs[j+1]|(((w.qh[g]>>(4*h+2))&3)<<8)],
                             iq2s_grid[w.qs[j]|(((w.qh[g]>>(4*h))&3)<<8)]);
    } else {
        const int j=8*g+4*h;const int qh=w.qh[g]>>(4*h);
        return _mm_set_epi32(iq3s_grid[w.qs[j+3]|(((qh>>3)&1)<<8)],iq3s_grid[w.qs[j+2]|(((qh>>2)&1)<<8)],
                             iq3s_grid[w.qs[j+1]|(((qh>>1)&1)<<8)],iq3s_grid[w.qs[j]|((qh&1)<<8)]);
    }
}
template<bool two> static inline const uint8_t * iq_sign_ptr(const iq_source<two> &w) {
    if constexpr(two)return w.qs+32;else return w.signs;
}
template<bool two> static inline int iq_scale(const uint8_t *scales,int g,int h) {
    if constexpr(two)return 2*((scales[g]>>(4*h))&15)+1;
    else return 2*((scales[g/2]>>(4*(g%2)))&15)+1;
}
template<bool two> static void iq_pack(const iq_source<two> &w,iq_compact_block &out) {
    out.d=ggml_fp16_to_fp32(w.d);std::memset(out.scales,0,8);std::memcpy(out.scales,w.scales,sizeof(w.scales));
    const auto *signs=iq_sign_ptr<two>(w);
    for(int g=0;g<8;++g) {
        __m128i codes[2];
        for(int h=0;h<2;++h) {
            const __m128i magnitude=iq_grid_half<two>(w,g,h);
            const __m128i sign=_mm_set_epi64x(iq_signs[signs[4*g+2*h+1]],iq_signs[signs[4*g+2*h]]);
            __m128i idx;
            if constexpr(two)idx=_mm_sub_epi8(_mm_setzero_si128(),_mm_add_epi8(_mm_cmpgt_epi8(magnitude,_mm_set1_epi8(8)),_mm_cmpgt_epi8(magnitude,_mm_set1_epi8(25))));
            else idx=_mm_and_si128(_mm_srli_epi16(_mm_sub_epi8(magnitude,_mm_set1_epi8(1)),1),_mm_set1_epi8(7));
            codes[h]=_mm_or_si128(idx,_mm_and_si128(_mm_cmpgt_epi8(_mm_setzero_si128(),sign),_mm_set1_epi8(8)));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out.qs+16*g),_mm_or_si128(codes[0],_mm_slli_epi16(codes[1],4)));
    }
}
template<bool two> __attribute__((noinline)) static float iq_compact_dot(int n,const iq_compact_block *w,const block_q8_K *x) {
    const __m128i lut=two?_mm_setr_epi8(8,25,43,0,0,0,0,0,-8,-25,-43,0,0,0,0,0):
                         _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-3,-5,-7,-9,-11,-13,-15);
    const __m128i mask=_mm_set1_epi8(15);__m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
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

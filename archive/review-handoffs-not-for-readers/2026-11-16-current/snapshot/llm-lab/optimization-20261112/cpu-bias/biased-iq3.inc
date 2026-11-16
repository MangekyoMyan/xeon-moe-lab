// Exact IQ3_XXS recoding with precomputed signed-weight correction.
struct alignas(16) biased_iq3_block {
    compact_block base;
    int16_t half_sum[8];
};
static_assert(sizeof(biased_iq3_block)==160,"biased IQ3 stride");
static void make_biased(const compact_block &src,biased_iq3_block &dst) {
    static const int lut[16]={4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62};
    dst.base=src;int sum[8]={};
    for(int g=0;g<8;++g)for(int j=0;j<16;++j){
        const uint8_t code=src.qs[16*g+j];
        sum[j/4]+=src.scale[g]*lut[code&15];
        sum[4+j/4]+=src.scale[g]*lut[code>>4];
    }
    for(int i=0;i<8;++i){
        if(sum[i]%2 || sum[i]/2 < -32768 || sum[i]/2 > 32767)throw std::runtime_error("IQ3 bias correction overflow");
        dst.half_sum[i]=int16_t(sum[i]/2);
    }
}
static bool contains_minus128(int n,const block_q8_K *x) {
    __m128i found=_mm_setzero_si128();const __m128i low=_mm_set1_epi8(-128);
    for(int b=0;b<n/256;++b)for(int g=0;g<16;++g)
        found=_mm_or_si128(found,_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16*g)),low));
    return _mm_movemask_epi8(found)!=0;
}
static void bias_activation(int n,const block_q8_K *x,block_q8_K *u) {
    std::memcpy(u,x,(n/256)*sizeof(block_q8_K));
    const __m128i flip=_mm_set1_epi8(-128);
    for(int b=0;b<n/256;++b)for(int g=0;g<16;++g)
        _mm_storeu_si128(reinterpret_cast<__m128i *>(u[b].qs+16*g),
            _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16*g)),flip));
}
// Mode0 is raw activation with XOR; mode1 is already biased; mode2 exactly
// preserves the previous PSIGNB behavior for external Q8_K containing -128.
template<int Mode> __attribute__((noinline)) static float biased_dot(int n,const biased_iq3_block *w,const block_q8_K *x) {
    const __m128i lut=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const __m128i mask=_mm_set1_epi8(15),flip=_mm_set1_epi8(-128);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b){
        __m128i low=_mm_setzero_si128(),high=low;
        #pragma GCC unroll 8
        for(int g=0;g<8;++g){
            const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w[b].base.qs+16*g));
            const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
            const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
            __m128i y0=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g));
            __m128i y1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+32*g+16));
            const __m128i scale=_mm_set1_epi16(w[b].base.scale[g]);
            if constexpr(Mode==2){
                low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q0),_mm_sign_epi8(y0,q0)),scale));
                high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(q1),_mm_sign_epi8(y1,q1)),scale));
            }else{
                if constexpr(Mode==0){y0=_mm_xor_si128(y0,flip);y1=_mm_xor_si128(y1,flip);}
                low=_mm_add_epi32(low,_mm_madd_epi16(_mm_maddubs_epi16(y0,q0),scale));
                high=_mm_add_epi32(high,_mm_madd_epi16(_mm_maddubs_epi16(y1,q1),scale));
            }
        }
        if constexpr(Mode!=2){
            const __m128i c0=_mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(w[b].half_sum)));
            const __m128i c1=_mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(w[b].half_sum+4)));
            low=_mm_sub_epi32(low,_mm_slli_epi32(c0,8));high=_mm_sub_epi32(high,_mm_slli_epi32(c1,8));
        }
        const __m256i ints=_mm256_insertf128_si256(_mm256_castsi128_si256(low),high,1);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_set1_ps(w[b].base.d*x[b].d),_mm256_cvtepi32_ps(ints)));
    }
    return reduce(acc);
}

// Exact IQ3_XXS recoding; arithmetic matches the pinned Ivy Bridge dot kernel.
struct alignas(16) expanded_block {
    int8_t q[256];
    float d;
    uint8_t scale[8];
};
static_assert(sizeof(expanded_block)==272,"expanded stride");
struct alignas(16) compact_block {
    uint8_t qs[128];
    float d;
    uint8_t scale[8];
};
static_assert(sizeof(compact_block)==144,"compact stride");

static inline float reduce(__m256 x) {
    __m128 sum=_mm_add_ps(_mm256_extractf128_ps(x,1),_mm256_castps256_ps128(x));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));
    return .25f*_mm_cvtss_f32(sum);
}

static void expand(const block_iq3_xxs *src,expanded_block *dst,size_t blocks) {
    static const auto signs64=[] {
        std::array<uint64_t,128> table{};
        for(int s=0;s<128;++s)for(int j=0;j<8;++j)
            table[s]|=uint64_t(ksigns_iq2xs[s]&(1<<j)?255:1)<<(8*j);
        return table;
    }();
    for(size_t b=0;b<blocks;++b) {
        dst[b].d=ggml_fp16_to_fp32(src[b].d);
        for(int g=0;g<8;++g) {
            uint32_t packed;std::memcpy(&packed,src[b].qs+64+4*g,4);
            dst[b].scale[g]=2*(packed>>28)+1;
            const uint8_t *q=src[b].qs+8*g;
            const __m128i low=_mm_set_epi32(iq3xxs_grid[q[3]],iq3xxs_grid[q[2]],iq3xxs_grid[q[1]],iq3xxs_grid[q[0]]);
            const __m128i high=_mm_set_epi32(iq3xxs_grid[q[7]],iq3xxs_grid[q[6]],iq3xxs_grid[q[5]],iq3xxs_grid[q[4]]);
            const __m128i sl=_mm_set_epi64x(signs64[(packed>>7)&127],signs64[packed&127]);
            const __m128i sh=_mm_set_epi64x(signs64[(packed>>21)&127],signs64[(packed>>14)&127]);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst[b].q+32*g),_mm_sign_epi8(low,sl));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst[b].q+32*g+16),_mm_sign_epi8(high,sh));
        }
    }
}

static void compact(const expanded_block &src,compact_block &dst) {
    dst.d=src.d;std::memcpy(dst.scale,src.scale,8);
    const auto encode=[](__m128i q) {
        const __m128i magnitude=_mm_sub_epi8(_mm_abs_epi8(q),_mm_set1_epi8(4));
        const __m128i index=_mm_and_si128(_mm_srli_epi16(magnitude,3),_mm_set1_epi8(7));
        return _mm_or_si128(index,_mm_and_si128(_mm_cmpgt_epi8(_mm_setzero_si128(),q),_mm_set1_epi8(8)));
    };
    for(int g=0;g<8;++g) {
        const __m128i lo=encode(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src.q+32*g)));
        const __m128i hi=encode(_mm_loadu_si128(reinterpret_cast<const __m128i *>(src.q+32*g+16)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst.qs+16*g),_mm_or_si128(lo,_mm_slli_epi16(hi,4)));
    }
}

__attribute__((noinline)) static float compact_dot(int n,const compact_block *w,const block_q8_K *x) {
    const __m128i lut=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const __m128i mask=_mm_set1_epi8(15);
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m128i low=_mm_setzero_si128(),high=low;
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


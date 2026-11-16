// Experimental single-row MXFP4 RHS reuse. Native weights and reduction order stay unchanged.
struct alignas(16) mx_correction { int16_t v[8]; };
struct mx_prepared {
    std::vector<float> scales;
    std::vector<mx_correction> correction;
    std::vector<float> broadcast;
    bool safe=true;
    explicit mx_prepared(size_t n):scales(n),correction(n),broadcast(4*n) {}
};

static void mx_prepare(int n,const block_q8_0 * x,mx_prepared & out,int mode) {
    const __m128i bias=_mm_set1_epi8(12), minimum=_mm_set1_epi8(-128);
    out.safe=true;
    for (int b=0;b<n/32;++b) {
        uint16_t bits;std::memcpy(&bits,&x[b].d,2);
        if ((bits&0x7c00)==0x7c00) out.safe=false;
        const __m128i a=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs));
        const __m128i c=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16));
        if (_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(a,minimum),_mm_cmpeq_epi8(c,minimum)))) out.safe=false;
        if (mode&2) _mm_store_si128(reinterpret_cast<__m128i *>(out.correction[b].v),
            _mm_add_epi16(_mm_maddubs_epi16(bias,a),_mm_maddubs_epi16(bias,c)));
        if (mode&1) out.scales[b]=GGML_CPU_FP16_TO_FP32(x[b].d);
        if (mode&4) _mm_storeu_ps(out.broadcast.data()+4*b,_mm_set1_ps(out.scales[b]));
    }
}

static inline float mx_hsum(__m256 x) {
    __m128 r=_mm256_extractf128_ps(x,1);
    r=_mm_add_ps(r,_mm256_castps256_ps128(x));
    r=_mm_add_ps(r,_mm_movehl_ps(r,r));
    r=_mm_add_ss(r,_mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

template<int mode> static inline __m128i mx_int_block(const block_mxfp4 & w,const block_q8_0 & x,const mx_correction & correction) {
    const __m128i mask=_mm_set1_epi8(15),ones=_mm_set1_epi16(1);
    const __m128i lut=(mode&2)?_mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0):
                                _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_fp4));
    const __m128i bits=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w.qs));
    const __m128i q0=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
    const __m128i q1=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
    const __m128i x0=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x.qs));
    const __m128i x1=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x.qs+16));
    if constexpr (mode&2) {
        const __m128i biased=_mm_add_epi16(_mm_maddubs_epi16(q0,x0),_mm_maddubs_epi16(q1,x1));
        return _mm_madd_epi16(_mm_sub_epi16(biased,_mm_load_si128(reinterpret_cast<const __m128i *>(correction.v))),ones);
    } else {
        const __m128i p0=_mm_maddubs_epi16(_mm_sign_epi8(q0,q0),_mm_sign_epi8(x0,q0));
        const __m128i p1=_mm_maddubs_epi16(_mm_sign_epi8(q1,q1),_mm_sign_epi8(x1,q1));
        return _mm_add_epi32(_mm_madd_epi16(p0,ones),_mm_madd_epi16(p1,ones));
    }
}

template<int mode> __attribute__((noinline)) static float mx_dot(int n,const block_mxfp4 * w,const block_q8_0 * x,const mx_prepared & rhs) {
    if (!rhs.safe) {float result;ggml_vec_dot_mxfp4_q8_0(n,&result,0,w,0,x,0,1);return result;}
    __m256 acc=_mm256_setzero_ps();int b=0;const int blocks=n/32;
    for (;b+1<blocks;b+=2) {
        const __m128i lo=mx_int_block<mode>(w[b],x[b],rhs.correction[b]);
        const __m128i hi=mx_int_block<mode>(w[b+1],x[b+1],rhs.correction[b+1]);
        const __m256 values=_mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(lo),hi,1));
        __m256 scales;
        if constexpr (mode&4) {
            const __m256 ws=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(w[b].e))),_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(w[b+1].e)),1);
            const __m256 xs=_mm256_loadu_ps(rhs.broadcast.data()+4*b);
            scales=_mm256_mul_ps(ws,xs);
        } else {
            const float d0=GGML_CPU_E8M0_TO_FP32_HALF(w[b].e)*((mode&1)?rhs.scales[b]:GGML_CPU_FP16_TO_FP32(x[b].d));
            const float d1=GGML_CPU_E8M0_TO_FP32_HALF(w[b+1].e)*((mode&1)?rhs.scales[b+1]:GGML_CPU_FP16_TO_FP32(x[b+1].d));
            scales=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(d0)),_mm_set1_ps(d1),1);
        }
        acc=_mm256_add_ps(_mm256_mul_ps(scales,values),acc);
    }
    float sum=mx_hsum(acc);
    // Keep the active reference's scalar odd-block tail, including multiplication order.
    for (;b<blocks;++b) {
        const float d=GGML_CPU_FP16_TO_FP32(x[b].d)*GGML_CPU_E8M0_TO_FP32_HALF(w[b].e);
        int s0=0,s1=0;
        for (int j=0;j<16;++j) {s0+=x[b].qs[j]*kvalues_fp4[w[b].qs[j]&15];s1+=x[b].qs[j+16]*kvalues_fp4[w[b].qs[j]>>4];}
        sum+=d*(s0+s1);
    }
    return sum;
}

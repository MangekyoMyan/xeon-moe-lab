#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "ggml-cpu/simd-mappings.h"
#include "mx-rhs.h"
#include <immintrin.h>
#include <cstring>

struct alignas(16) mx_correction { int16_t v[8]; };
struct mx_rhs_view {
    const float * scales;
    const mx_correction * correction;
    const float * broadcast;
    bool safe;
};
#include "mx-math.inc"

int xeon_mxfp4_rhs_prepare_v1(int n,const void * input,int first,int end,void * sidecar) {
    GGML_ASSERT(n>0 && n%32==0 && first>=0 && first<=end && end<=n/32);
    const auto * x=static_cast<const block_q8_0 *>(input);
    auto * correction=static_cast<mx_correction *>(sidecar);
    auto * scales=reinterpret_cast<float *>(correction+n/32);
    const __m128i bias=_mm_set1_epi8(12),minimum=_mm_set1_epi8(-128);
    bool safe=true;
    for(int b=first;b<end;++b) {
        uint16_t bits;std::memcpy(&bits,&x[b].d,2);
        if((bits&0x7c00)==0x7c00)safe=false;
        const __m128i a=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs));
        const __m128i c=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16));
        if(_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(a,minimum),_mm_cmpeq_epi8(c,minimum))))safe=false;
        _mm_store_si128(reinterpret_cast<__m128i *>(correction[b].v),_mm_add_epi16(_mm_maddubs_epi16(bias,a),_mm_maddubs_epi16(bias,c)));
        scales[b]=GGML_CPU_FP16_TO_FP32(x[b].d);
    }
    return safe?1:0;
}

void xeon_mxfp4_rhs_dot_v1(int n,float * output,const void * weights,const void * input,const void * sidecar) {
    const auto * correction=static_cast<const mx_correction *>(sidecar);
    const mx_rhs_view view{reinterpret_cast<const float *>(correction+n/32),correction,nullptr,true};
    *output=mx_dot<3>(n,static_cast<const block_mxfp4 *>(weights),static_cast<const block_q8_0 *>(input),view);
}

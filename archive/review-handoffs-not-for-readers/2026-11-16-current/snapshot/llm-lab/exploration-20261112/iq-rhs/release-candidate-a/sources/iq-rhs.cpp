#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "iq-rhs.h"
#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstring>
#include <cmath>
#include "compact-kernel.inc"
struct alignas(16) rhs_correction { int16_t pairs[16][8]; };
static_assert(sizeof(rhs_correction)==256,"correction stride");
#include "iq-math.inc"
int xeon_iq3_rhs_prepare_v1(int n,const void * input,int first,int end,void * output) {
    GGML_ASSERT(n>0 && n%256==0 && first>=0 && first<=end && end<=n/256);
    const auto * x=static_cast<const block_q8_K *>(input);
    auto * correction=static_cast<rhs_correction *>(output);
    const __m128i bias=_mm_set1_epi8(62),minimum=_mm_set1_epi8(-128);bool safe=true;
    for(int b=first;b<end;++b){
        if(!std::isfinite(x[b].d))safe=false;
        for(int h=0;h<16;++h){
            const __m128i values=_mm_loadu_si128(reinterpret_cast<const __m128i *>(x[b].qs+16*h));
            if(_mm_movemask_epi8(_mm_cmpeq_epi8(values,minimum)))safe=false;
            _mm_store_si128(reinterpret_cast<__m128i *>(correction[b].pairs[h]),_mm_maddubs_epi16(bias,values));
        }
    }
    return safe?1:0;
}
void xeon_iq3_rhs_dot_v1(int n,float * output,const void * weights,const void * input,const void * correction) {
    *output=prepared_dot(n,static_cast<const compact_block *>(weights),static_cast<const block_q8_K *>(input),static_cast<const rhs_correction *>(correction));
}

// Optional test-only interposition. No performance measurements use this DSO.
#include "ggml-cpu.h"
#include <atomic>
#include <cstdint>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

static std::atomic<uint64_t> counts[8]{};
#define COUNT_DOT(name, index) \
extern "C" void name(int n, float * s, size_t bs, const void * w, size_t bw, const void * x, size_t bx, int nc) { \
    static auto original = reinterpret_cast<ggml_vec_dot_t>(dlsym(RTLD_NEXT,#name)); \
    if (!original) std::abort(); \
    counts[index].fetch_add(1,std::memory_order_relaxed); original(n,s,bs,w,bw,x,bx,nc); \
}
COUNT_DOT(ggml_vec_dot_iq3_xxs_q8_K,0)
COUNT_DOT(ggml_vec_dot_iq3_s_q8_K,0)
COUNT_DOT(ggml_vec_dot_iq2_s_q8_K,0)
COUNT_DOT(xeon_iq3_compact_dot,1)
COUNT_DOT(xeon_iq2s_compact_dot,1)
COUNT_DOT(xeon_iq3s_compact_dot,1)
extern "C" ggml_vec_dot_t xeon_iq_input_dot(ggml_type t) {
    using Select = ggml_vec_dot_t (*)(ggml_type);
    static auto original = reinterpret_cast<Select>(dlsym(RTLD_NEXT,"xeon_iq_input_dot"));
    if (!original) std::abort();
    counts[2].fetch_add(1,std::memory_order_relaxed); return original(t);
}
extern "C" void xeon_iq_input_from_q8(const void * x, void * y, size_t n) {
    using Pack = void (*)(const void *, void *, size_t);
    static auto original = reinterpret_cast<Pack>(dlsym(RTLD_NEXT,"xeon_iq_input_from_q8"));
    if (!original) std::abort();
    counts[3].fetch_add(n,std::memory_order_relaxed);
    counts[4].fetch_add(1,std::memory_order_relaxed);
    const int mode=std::fegetround();
    const unsigned bit=mode==FE_TONEAREST ? 1 : mode==FE_DOWNWARD ? 2 : mode==FE_UPWARD ? 4 : mode==FE_TOWARDZERO ? 8 : 16;
    counts[7].fetch_or(bit,std::memory_order_relaxed);
    original(x,y,n);
}
extern "C" int xeon_mxfp4_rhs_prepare_v1(int n,const void * x,int first,int end,void * out) {
    using Fn=int (*)(int,const void *,int,int,void *);
    static auto original=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"xeon_mxfp4_rhs_prepare_v1"));
    if (!original) std::abort();
    counts[5].fetch_add(1,std::memory_order_relaxed);
    return original(n,x,first,end,out);
}
extern "C" void xeon_mxfp4_rhs_dot_v1(int n,float * out,const void * w,const void * x,const void * rhs) {
    using Fn=void (*)(int,float *,const void *,const void *,const void *);
    static auto original=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"xeon_mxfp4_rhs_dot_v1"));
    if (!original) std::abort();
    counts[6].fetch_add(1,std::memory_order_relaxed);
    original(n,out,w,x,rhs);
}
extern "C" void test_reset_counts(uint64_t * values) {
    for (unsigned i=0;i<8;++i) values[i]=counts[i].exchange(0,std::memory_order_relaxed);
}

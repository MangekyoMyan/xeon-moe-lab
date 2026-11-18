// Optional test-only interposition. No performance measurements use this DSO.
#include "ggml-cpu.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

static std::atomic<uint64_t> counts[4]{};
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
    counts[3].fetch_add(n,std::memory_order_relaxed); original(x,y,n);
}
__attribute__((destructor)) static void finish() {
    const char * path = std::getenv("XEON_TEST_COUNTS");
    if (!path) return;
    FILE * f = std::fopen(path,"w");
    if (!f) std::abort();
    std::fprintf(f,"native_row_dots,legacy_compact_row_dots,folded_selector_calls,q8_blocks_prepared\n%llu,%llu,%llu,%llu\n",
        (unsigned long long)counts[0].load(),(unsigned long long)counts[1].load(),
        (unsigned long long)counts[2].load(),(unsigned long long)counts[3].load());
    std::fclose(f);
}

#include "ggml.h"
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <immintrin.h>
#include <stdexcept>

struct Q8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
struct alignas(16) CompactXXS {
    uint8_t qs[128];
    float d;
    uint8_t scale[8];
};
static_assert(sizeof(Q8K) == 292);
static_assert(sizeof(CompactXXS) == 144);
using Quant = void (*)(const float *, void *, int64_t);
using Dot = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
using SelectDot = Dot (*)(enum ggml_type);

template<class F> F sym(void * handle, const char * name) {
    void * p = dlsym(handle, name);
    if (!p) throw std::runtime_error(name);
    return reinterpret_cast<F>(p);
}
static uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    void * h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "%s\n", dlerror()); return 3; }
    auto quant = sym<Quant>(h, "quantize_row_q8_K");
    auto ref = sym<Quant>(h, "quantize_row_q8_K_ref");
    auto prepare = sym<Quant>(h, "xeon_iq_input_from_float");
    auto old_dot = sym<Dot>(h, "xeon_iq3_compact_dot");
    auto select = sym<SelectDot>(h, "xeon_iq_input_dot");
    auto new_dot = select(GGML_TYPE_IQ3_XXS);
    alignas(32) float x[256] = {};
    uint32_t tenth = 0x3dcccccd;
    std::memcpy(x, &tenth, sizeof(tenth));
    CompactXXS w{};
    std::memset(w.qs, 0x88, sizeof(w.qs));
    w.d = 1.f;
    std::memset(w.scale, 1, sizeof(w.scale));
    std::puts("fenv_mode,mxcsr_round,quant_ref_bytes_equal,q0,d_hex,old_output,new_output,old_bits,new_bits,output_bytes_equal");
    for (int rm : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        if (std::fesetround(rm)) return 4;
        Q8K q, qr;
        std::memset(&q, 0xa5, sizeof(q));
        std::memset(&qr, 0xa5, sizeof(qr));
        alignas(16) uint8_t folded[400];
        std::memset(folded, 0xa5, sizeof(folded));
        quant(x, &q, 256);
        ref(x, &qr, 256);
        prepare(x, folded, 256);
        float a = 0.f, b = 0.f;
        old_dot(256, &a, 0, &w, 0, &q, 0, 1);
        new_dot(256, &b, 0, &w, 0, folded, 0, 1);
        std::printf("%d,%u,%d,%d,%08x,%.9g,%.9g,%08x,%08x,%d\n", rm,
            unsigned(_MM_GET_ROUNDING_MODE()), std::memcmp(&q,&qr,sizeof(q))==0,
            int(q.qs[0]), bits(q.d), double(a), double(b), bits(a), bits(b), bits(a)==bits(b));
    }
    std::fesetround(FE_TONEAREST);
    return 0;
}

// Test adapters for the complete functions in the supplied diagnostic excerpts.
#include "iq3-cache.h"
#include "mx-rhs.h"
#include "route37.h"
#include <cstdlib>
#include <vector>

extern "C" const void * const * xeon_iq3_cache_lookup_experts_v2(const ggml_tensor * t) {
    const char * base = static_cast<const char *>(xeon_iq3_cache_lookup(t));
    if (!base) return nullptr;
    const char * setting = std::getenv("XEON_TEST_MASK");
    const uint64_t mask = setting ? std::strtoull(setting, nullptr, 0) : UINT64_MAX;
    const size_t stride = (size_t(t->ne[0])/256)*t->ne[1]*144;
    thread_local std::vector<const void *> pointers;
    pointers.resize(t->ne[2]);
    for (int64_t e=0;e<t->ne[2];++e) pointers[e] = ((mask >> (e%64))&1) ? base+size_t(e)*stride : nullptr;
    return pointers.data();
}
extern "C" void xeon_route37_record_v1(const ggml_tensor *, const ggml_tensor *) {}
extern "C" int xeon_mxfp4_rhs_prepare_v1(int, const void *, int, int, void *) { std::abort(); }
extern "C" void xeon_mxfp4_rhs_dot_v1(int, float *, const void *, const void *, const void *) { std::abort(); }

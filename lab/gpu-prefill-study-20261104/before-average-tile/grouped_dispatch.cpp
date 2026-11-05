#include "ggml.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>

// Experimental, standalone-process-only interposition. The production API is untouched.
static std::atomic<int> mode{0};
static std::atomic<uint64_t> forced{0};
extern "C" void xeon_grouped_mode(int value) { mode.store(value); }
extern "C" uint64_t xeon_grouped_forced_calls() { return forced.load(); }

bool ggml_cuda_should_use_mmq(ggml_type type, int cc, int64_t columns, int64_t experts) {
    using Fn = bool (*)(ggml_type, int, int64_t, int64_t);
    static const auto original = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "_Z24ggml_cuda_should_use_mmq9ggml_typeill"));
    if (!original) std::abort();
    const bool tested_type = type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS ||
                             type == GGML_TYPE_IQ3_S || type == GGML_TYPE_MXFP4;
    if (mode.load() && cc == 700 && experts > 0 && columns >= 64 && tested_type) {
        ++forced;
        return true;
    }
    return original(type, cc, columns, experts);
}

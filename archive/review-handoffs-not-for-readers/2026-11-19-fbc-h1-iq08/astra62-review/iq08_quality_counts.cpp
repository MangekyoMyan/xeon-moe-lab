// Optional LD_PRELOAD library for a fresh 08 numerical/path-check process.
#include "iq08_quality_counts.h"
#include "ggml-cpu.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#define IQ08_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
using pack_fn = void (*)(const void *, void *, size_t);
using selector_fn = ggml_vec_dot_t (*)(enum ggml_type);

struct type_state {
    std::atomic<ggml_vec_dot_t> target{nullptr};
    std::atomic<uint64_t> selectors{0};
    std::atomic<uint64_t> rows{0};
    std::atomic<uint64_t> other_nrc{0};
};

std::array<type_state, IQ08_QUALITY_TYPE_COUNT> by_type{};
std::atomic<pack_fn> observed_pack{nullptr};
std::atomic<selector_fn> observed_selector{nullptr};
std::atomic<uint64_t> pack_calls{0};
std::atomic<uint64_t> pack_blocks{0};
std::atomic<uint64_t> empty_pack_calls{0};
std::atomic<uint64_t> other_selector_calls{0};

[[noreturn]] void fail(const char *reason) {
    std::fprintf(stderr, "08 quality counter: %s\n", reason);
    std::abort();
}

// Resolve only inside the corresponding hook. The ordinary 33a9 CPU library
// exports neither new function and can be used for a zero-count control.
template <typename Fn>
Fn resolve_next(const char *symbol, std::atomic<Fn> &receipt) {
    void *address = dlsym(RTLD_NEXT, symbol);
    if (!address) {
        fail(symbol);
    }
    auto target = reinterpret_cast<Fn>(address);
    receipt.store(target, std::memory_order_release);
    return target;
}

template <size_t Index>
void count_dot(int n, float *s, size_t bs, const void *x, size_t bx,
               const void *y, size_t by, int nrc) {
    auto &state = by_type[Index];
    ggml_vec_dot_t target = state.target.load(std::memory_order_acquire);
    if (!target) {
        fail("folded dot wrapper called before its selector");
    }
    target(n, s, bs, x, bx, y, by, nrc);
    if (nrc == 1) {
        state.rows.fetch_add(1, std::memory_order_relaxed);
    } else {
        state.other_nrc.fetch_add(1, std::memory_order_relaxed);
    }
}

template <size_t Index>
ggml_vec_dot_t wrap_dot(ggml_vec_dot_t actual) {
    // The pinned selector returns one stable function per type. Verify that
    // contract across workers so this wrapper always forwards the exact result.
    auto &state = by_type[Index];
    ggml_vec_dot_t seen = nullptr;
    if (!state.target.compare_exchange_strong(seen, actual, std::memory_order_acq_rel,
                                              std::memory_order_acquire) && seen != actual) {
        fail("selector changed its returned function for one ggml type");
    }
    state.selectors.fetch_add(1, std::memory_order_relaxed);
    return &count_dot<Index>;
}

template <typename Fn>
bool describe_binding(Fn actual, iq08_quality_binding_v1 &binding) {
    binding = {};
    if (!actual) {
        return true;
    }
    void *address = reinterpret_cast<void *>(actual);
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_fname || !info.dli_fbase) {
        return false;
    }
    binding.dso_path = info.dli_fname;
    binding.offset_from_dso_base = reinterpret_cast<uintptr_t>(address) -
                                   reinterpret_cast<uintptr_t>(info.dli_fbase);
    return true;
}
}  // namespace

IQ08_EXPORT void xeon_iq_input_from_q8(const void *input, void *output, size_t count) {
    static const auto target = resolve_next("xeon_iq_input_from_q8", observed_pack);
    target(input, output, count);
    pack_calls.fetch_add(1, std::memory_order_relaxed);
    pack_blocks.fetch_add(count, std::memory_order_relaxed);
    if (count == 0) {
        empty_pack_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

IQ08_EXPORT ggml_vec_dot_t xeon_iq_input_dot(enum ggml_type type) {
    static const auto selector = resolve_next("xeon_iq_input_dot", observed_selector);
    ggml_vec_dot_t actual = selector(type);
    if (!actual) {
        fail("folded selector returned a null function");
    }
    switch (type) {
        case GGML_TYPE_IQ2_S:   return wrap_dot<IQ08_QUALITY_IQ2_S>(actual);
        case GGML_TYPE_IQ3_XXS: return wrap_dot<IQ08_QUALITY_IQ3_XXS>(actual);
        case GGML_TYPE_IQ3_S:   return wrap_dot<IQ08_QUALITY_IQ3_S>(actual);
        default:
            other_selector_calls.fetch_add(1, std::memory_order_relaxed);
            return actual;
    }
}

IQ08_EXPORT int xeon_iq08_quality_snapshot_v1(iq08_quality_counts_v1 *out, size_t out_bytes) {
    if (!out || out_bytes != sizeof(*out)) {
        return 0;
    }
    iq08_quality_counts_v1 next{};
    next.abi_version = 1;
    next.struct_bytes = sizeof(next);
    next.pack_calls = pack_calls.load(std::memory_order_relaxed);
    next.pack_blocks = pack_blocks.load(std::memory_order_relaxed);
    next.empty_pack_calls = empty_pack_calls.load(std::memory_order_relaxed);
    next.other_selector_calls = other_selector_calls.load(std::memory_order_relaxed);
    if (!describe_binding(observed_pack.load(std::memory_order_acquire), next.underlying_pack) ||
        !describe_binding(observed_selector.load(std::memory_order_acquire), next.underlying_selector)) {
        return 0;
    }
    for (size_t i = 0; i < by_type.size(); ++i) {
        const auto &state = by_type[i];
        next.type[i].selector_calls = state.selectors.load(std::memory_order_relaxed);
        next.type[i].dot_rows_nrc1 = state.rows.load(std::memory_order_relaxed);
        next.type[i].other_nrc_calls = state.other_nrc.load(std::memory_order_relaxed);
        if (!describe_binding(state.target.load(std::memory_order_acquire), next.type[i].underlying_dot)) {
            return 0;
        }
    }
    *out = next;
    return 1;
}

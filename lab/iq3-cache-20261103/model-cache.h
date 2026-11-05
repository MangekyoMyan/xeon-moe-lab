#pragma once
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace xeon_iq3 {
struct cache {
    void * owner = nullptr;
    using create_fn = void * (*)(size_t, size_t, int);
    using add_fn = int (*)(void *, const ggml_tensor *);
    using free_fn = void (*)(void *);
    using used_fn = size_t (*)(const void *);
    add_fn add = nullptr;
    free_fn release = nullptr;
    used_fn used = nullptr;
    cache() = default;
    cache(const cache &) = delete;
    cache & operator=(const cache &) = delete;
    ~cache() { clear(); }
    void clear() { if (owner) { release(owner); owner = nullptr; } }
    void init(size_t bytes) {
        auto reg = ggml_backend_reg_by_name("CPU");
        if (!reg) throw std::runtime_error("IQ3 cache: CPU backend unavailable");
        auto create = reinterpret_cast<create_fn>(ggml_backend_reg_get_proc_address(reg, "xeon_iq3_cache_create_v1"));
        add = reinterpret_cast<add_fn>(ggml_backend_reg_get_proc_address(reg, "xeon_iq3_cache_add_v1"));
        release = reinterpret_cast<free_fn>(ggml_backend_reg_get_proc_address(reg, "xeon_iq3_cache_free_v1"));
        used = reinterpret_cast<used_fn>(ggml_backend_reg_get_proc_address(reg, "xeon_iq3_cache_used_v1"));
        if (!create || !add || !release || !used) throw std::runtime_error("IQ3 cache: matching CPU overlay required");
        owner = create(bytes, size_t(64)<<30, 12);
        if (!owner) throw std::runtime_error("IQ3 cache: owner allocation failed");
    }
};
}

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xeon_expert_dma {

using range = std::pair<size_t, size_t>; // [begin, end)
using register_fn = bool (*)(void *, size_t);
using unregister_fn = void (*)(void *);

inline std::vector<range> page_ranges(std::vector<range> ranges, size_t file_size, size_t page_size) {
    if (page_size == 0) throw std::runtime_error("invalid page size");
    std::vector<range> aligned;
    for (const auto & r : ranges) {
        if (r.first > r.second || r.second > file_size) throw std::runtime_error("expert range outside mmap");
        if (r.first == r.second) continue;
        const size_t start = r.first - r.first % page_size;
        const size_t remainder = r.second % page_size;
        const size_t extra = remainder ? page_size - remainder : 0;
        const size_t end = r.second + std::min(extra, file_size-r.second);
        aligned.emplace_back(start, end);
    }
    std::sort(aligned.begin(), aligned.end());
    std::vector<range> merged;
    for (const auto & r : aligned) {
        if (merged.empty() || r.first > merged.back().second) merged.push_back(r);
        else merged.back().second = std::max(merged.back().second, r.second);
    }
    return merged;
}

class registrations {
    struct entry { void * base; unregister_fn release; };
    std::vector<entry> entries;
public:
    registrations() = default;
    registrations(const registrations &) = delete;
    registrations & operator=(const registrations &) = delete;
    void reserve(size_t count) { entries.reserve(count); }
    bool add(void * base, size_t bytes, register_fn acquire, unregister_fn release) {
        if (!base || !bytes || !acquire || !release) throw std::runtime_error("invalid registration");
        if (!acquire(base, bytes)) return false;
        try { entries.push_back({base, release}); }
        catch (...) { release(base); throw; }
        return true;
    }
    void clear() noexcept {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) it->release(it->base);
        entries.clear();
    }
    ~registrations() { clear(); }
};

} // namespace xeon_expert_dma

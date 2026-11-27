#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include "iq3-cache.h"
#include "ggml-backend.h"
#include <immintrin.h>
#include <omp.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <type_traits>
#include <sys/mman.h>
#include <unistd.h>

#include "compact-kernel.inc"
#include "iq-small-kernels.inc"
static_assert(sizeof(compact_block)==sizeof(iq_compact_block), "shared compact stride");

namespace {
constexpr size_t cap = size_t(128) << 30;
struct entry {
    const void * source = nullptr;
    void * data = MAP_FAILED;
    size_t bytes = 0;
    enum ggml_type type;
    int64_t ne[4];
    size_t nb[4];
    std::vector<const void *> expert_ptrs;
    bool legacy_identity = false;
    ~entry() { if (data != MAP_FAILED) munmap(data, bytes); }
};
struct owner {
    size_t budget, reserve, used = 0;
    int threads;
    std::vector<std::unique_ptr<entry>> entries;
};
// Model loading and destruction take this lock. Read lookup takes it once per worker/operator.
std::mutex registry_mutex;
std::unordered_map<const void *, entry *> registry;
size_t allocated = 0;
std::atomic<uint64_t> hits{0};

size_t available_memory() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        unsigned long long kb;
        if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1 &&
            kb <= std::numeric_limits<size_t>::max()/1024) return size_t(kb)*1024;
    }
    return 0;
}
bool eligible(const ggml_tensor * t) {
    return t && (t->type == GGML_TYPE_IQ3_XXS || t->type == GGML_TYPE_IQ2_S || t->type == GGML_TYPE_IQ3_S) && t->data && t->buffer &&
        ggml_backend_buffer_is_host(t->buffer) &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        t->ne[0] > 0 && t->ne[0] <= INT32_MAX && t->ne[0]%256 == 0 &&
        t->ne[1] > 0 && t->ne[2] > 0 && t->ne[3] == 1 && ggml_is_contiguous(t);
}
}

void * xeon_iq3_cache_create_v1(size_t budget, size_t reserve, int threads) {
    if (!budget || budget > cap || threads < 1 || threads > 12) return nullptr;
    try { return new owner{budget, reserve, 0, threads, {}}; }
    catch (const std::exception & error) { std::fprintf(stderr,"IQ cache owner: %s\n",error.what());return nullptr; }
    catch (...) { return nullptr; }
}

static int add_selected(void * opaque, const ggml_tensor * t, const uint32_t * selected, size_t selected_count, bool full_identity) {
    // 1 = added, 0 = ineligible/budget/duplicate, -1 = allocation or reserve failure.
    if (!opaque || !eligible(t) || t->ne[2] > 65536) return 0;
    try {
        auto & o = *static_cast<owner *>(opaque);
        const size_t experts = size_t(t->ne[2]);
        if (full_identity) selected_count = experts;
        if (!selected_count || selected_count > experts || (!full_identity && !selected)) return 0;
        const size_t original_blocks = ggml_nbytes(t)/ggml_type_size(t->type);
        if (original_blocks%experts) return 0;
        const size_t expert_blocks = original_blocks/experts;
        if (expert_blocks > std::numeric_limits<size_t>::max()/selected_count) return 0;
        const size_t blocks = expert_blocks*selected_count;
        if (blocks > (std::numeric_limits<size_t>::max()-65536)/sizeof(compact_block)) return 0;
        std::vector<uint32_t> ids(selected_count);
        std::vector<uint8_t> seen(experts, 0);
        bool identity = selected_count == experts;
        for (size_t slot = 0; slot < selected_count; ++slot) {
            const uint32_t id = full_identity ? uint32_t(slot) : selected[slot];
            if (id >= experts || seen[id]) return 0;
            seen[id] = 1; ids[slot] = id;
            identity = identity && id == slot;
        }
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || page > 65536) return -1;
        const size_t bytes = (blocks*sizeof(compact_block)+page-1)/page*page;
        std::unique_lock<std::mutex> lock(registry_mutex);
        if (registry.count(t->data) || bytes > o.budget-o.used || bytes > cap-allocated) return 0;
        const size_t available = available_memory();
        if (available < o.reserve || bytes > available-o.reserve) return -1;
        auto e = std::make_unique<entry>();
        e->source = t->data;
        e->type = t->type;
        e->bytes = bytes;
        std::memcpy(e->ne, t->ne, sizeof(e->ne));
        std::memcpy(e->nb, t->nb, sizeof(e->nb));
        e->expert_ptrs.resize(experts, nullptr);
        e->legacy_identity = identity;
        e->data = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (e->data == MAP_FAILED) return -1;
        // Allocate metadata before touching the large buffer. No publication until it is complete.
        o.entries.reserve(o.entries.size()+1);
        registry.reserve(registry.size()+1);
        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for (size_t slot = 0; slot < selected_count; ++slot) {
            const size_t source_offset = size_t(ids[slot])*expert_blocks;
            const size_t target_offset = slot*expert_blocks;
            for (size_t b = 0; b < expert_blocks; ++b) {
                if (t->type == GGML_TYPE_IQ3_XXS) {
                    expanded_block temp;
                    expand(static_cast<const block_iq3_xxs *>(t->data)+source_offset+b, &temp, 1);
                    compact(temp, static_cast<compact_block *>(e->data)[target_offset+b]);
                } else if (t->type == GGML_TYPE_IQ2_S) {
                    iq_pack<true>(static_cast<const block_iq2_s *>(t->data)[source_offset+b], static_cast<iq_compact_block *>(e->data)[target_offset+b]);
                } else {
                    iq_pack<false>(static_cast<const block_iq3_s *>(t->data)[source_offset+b], static_cast<iq_compact_block *>(e->data)[target_offset+b]);
                }
            }
        }
        for (size_t slot = 0; slot < selected_count; ++slot) {
            e->expert_ptrs[ids[slot]] = static_cast<const char *>(e->data) + slot*expert_blocks*sizeof(compact_block);
        }
        registry.emplace(e->source, e.get());
        o.entries.push_back(std::move(e));
        o.used += bytes;
        allocated += bytes;
        return 1;
    } catch (...) { return -1; }
}

int xeon_iq3_cache_add_v1(void * owner, const ggml_tensor * tensor) {
    return add_selected(owner, tensor, nullptr, 0, true);
}

int xeon_iq3_cache_add_selected_v2(void * owner, const ggml_tensor * tensor, const uint32_t * experts, size_t count) {
    return add_selected(owner, tensor, experts, count, false);
}

void xeon_iq3_cache_free_v1(void * opaque) {
    if (!opaque) return;
    auto * o = static_cast<owner *>(opaque);
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (const auto & e : o->entries) registry.erase(e->source);
    allocated -= o->used;
    delete o;
}

size_t xeon_iq3_cache_used_v1(const void * opaque) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    return opaque ? static_cast<const owner *>(opaque)->used : 0;
}

const void * xeon_iq3_cache_lookup(const ggml_tensor * t) {
    if (!eligible(t)) return nullptr;
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = registry.find(t->data);
    if (found == registry.end()) return nullptr;
    const entry & e = *found->second;
    if (e.type != t->type || std::memcmp(e.ne, t->ne, sizeof(e.ne)) || std::memcmp(e.nb, t->nb, sizeof(e.nb)) || !e.legacy_identity) return nullptr;
    hits.fetch_add(1, std::memory_order_relaxed);
    return e.data;
}

const void * const * xeon_iq3_cache_lookup_experts_v2(const ggml_tensor * t) {
    if (!eligible(t)) return nullptr;
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = registry.find(t->data);
    if (found == registry.end()) return nullptr;
    const entry & e = *found->second;
    if (e.type != t->type || std::memcmp(e.ne, t->ne, sizeof(e.ne)) || std::memcmp(e.nb, t->nb, sizeof(e.nb))) return nullptr;
    hits.fetch_add(1, std::memory_order_relaxed);
    return e.expert_ptrs.data();
}

uint64_t xeon_iq3_cache_hits_v1() { return hits.load(std::memory_order_relaxed); }
size_t xeon_iq3_cache_total_v1() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    return allocated;
}

void xeon_iq3_compact_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1 && n%256 == 0);
    *s = compact_dot(n, static_cast<const compact_block *>(x), static_cast<const block_q8_K *>(y));
    (void) bs; (void) bx; (void) by;
}

void xeon_iq2s_compact_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1 && n%256 == 0);
    *s = iq_compact_dot<true>(n, static_cast<const iq_compact_block *>(x), static_cast<const block_q8_K *>(y));
    (void) bs; (void) bx; (void) by;
}
void xeon_iq3s_compact_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1 && n%256 == 0);
    *s = iq_compact_dot<false>(n, static_cast<const iq_compact_block *>(x), static_cast<const block_q8_K *>(y));
    (void) bs; (void) bx; (void) by;
}

#include "census.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <limits>
#include <new>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
constexpr uint32_t capacity = 128, max_experts = 512;
struct alignas(64) header {
    char magic[8];
    uint32_t version, record_bytes, capacity, pid;
    uint64_t failure, used, dropped;
    char padding[16];
};
struct alignas(64) record {
    uint64_t sequence;
    char name[64];
    int32_t type;
    uint32_t experts;
    int64_t k, rows;
    uint64_t calls, assignments;
    uint64_t counts[max_experts];
};
struct mapped_stats { header h; record records[capacity]; };
static_assert(sizeof(header) == 64, "header ABI");
static_assert(sizeof(record) == 4224, "record ABI");
static_assert(offsetof(record, name) == 8 && offsetof(record, type) == 72 && offsetof(record, experts) == 76, "record prefix ABI");
static_assert(offsetof(record, k) == 80 && offsetof(record, rows) == 88 && offsetof(record, calls) == 96, "record shape ABI");
static_assert(offsetof(record, assignments) == 104 && offsetof(record, counts) == 112, "record counters ABI");

struct collector {
    mapped_stats * data = nullptr;
    std::mutex mutex;
    std::unordered_map<const void *, size_t> slots;
    collector() {
        const char * path = std::getenv("XEON_MOE_CENSUS_PATH");
        if (!path || path[0] != '/') return;
        const int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, 0600);
        if (fd < 0) { std::fprintf(stderr, "xeon census: cannot create exclusive aggregate file\n"); return; }
        if (ftruncate(fd, sizeof(mapped_stats))) { close(fd); return; }
        void * memory = mmap(nullptr, sizeof(mapped_stats), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (memory == MAP_FAILED) return;
        data = new(memory) mapped_stats{};
        std::memcpy(data->h.magic, "XMCENS01", 8);
        data->h.version = 1;
        data->h.record_bytes = sizeof(record);
        data->h.capacity = capacity;
        data->h.pid = getpid();
        std::fprintf(stderr, "xeon census: single-token aggregate collection enabled\n");
    }
};

collector & get_collector() { static collector c; return c; }
std::atomic<bool> broken{false};
}

void xeon_moe_census_v1(const ggml_tensor * weights, const ggml_tensor * ids, const int64_t * counts, int64_t experts) {
    if (broken.load(std::memory_order_relaxed)) return;
    collector * state = nullptr;
    try {
    if (!weights || !ids || ids->ne[1] != 1 || !weights->data || !counts || experts <= 0 || experts > max_experts ||
        weights->ne[2] != experts || std::strncmp(weights->name, "blk.", 4) || !std::strstr(weights->name, "_exps.weight")) return;
    collector & c = get_collector();
    state = &c;
    if (!c.data) return;
    std::lock_guard<std::mutex> lock(c.mutex);
    size_t slot;
    const auto found = c.slots.find(weights->data);
    if (found == c.slots.end()) {
        slot = c.data->h.used;
        if (slot >= capacity) { ++c.data->h.dropped; return; }
        auto & r = c.data->records[slot];
        std::memcpy(r.name, weights->name, std::min(sizeof(r.name)-1, sizeof(weights->name)));
        r.type = weights->type; r.experts = experts; r.k = weights->ne[0]; r.rows = weights->ne[1];
        c.slots.emplace(weights->data, slot);
        __atomic_store_n(&c.data->h.used, slot+1, __ATOMIC_RELEASE);
    } else slot = found->second;
    auto & r = c.data->records[slot];
    if (r.type != weights->type || r.experts != experts || r.k != weights->ne[0] || r.rows != weights->ne[1] ||
        std::strncmp(r.name, weights->name, sizeof(r.name))) { ++c.data->h.dropped; return; }
    if (ids->ne[0] < 0) { ++c.data->h.dropped; return; }
    const uint64_t limit = uint64_t(ids->ne[0]);
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    uint64_t total = 0;
    for (int i = 0; i < experts; ++i) {
        if (counts[i] < 0 || uint64_t(counts[i]) > limit-total || uint64_t(counts[i]) > maximum-r.counts[i]) {
            ++c.data->h.dropped; return;
        }
        total += uint64_t(counts[i]);
    }
    if (r.calls == maximum || total > maximum-r.assignments || r.sequence > maximum-2) { ++c.data->h.dropped; return; }
    __atomic_add_fetch(&r.sequence, 1, __ATOMIC_SEQ_CST);
    ++r.calls;
    for (int i = 0; i < experts; ++i) r.counts[i] += counts[i];
    r.assignments += total;
    __atomic_add_fetch(&r.sequence, 1, __ATOMIC_RELEASE);
    } catch (...) {
        broken.store(true, std::memory_order_relaxed);
        if (state && state->data) __atomic_store_n(&state->data->h.failure, 1, __ATOMIC_RELEASE);
        std::fprintf(stderr, "xeon census: disabled after collection failure\n");
    }
}

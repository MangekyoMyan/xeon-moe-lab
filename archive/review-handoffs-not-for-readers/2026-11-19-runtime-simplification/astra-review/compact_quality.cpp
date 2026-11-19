// Isolated CPU graph and anonymous-memory checks for the current interleave profile.
// Only synthetic buffers created in this process receive deliberately changed weights.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "iq3-cache.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static void require(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}

static void check_provider(const char *name, const char *expected) {
    Dl_info info{};
    void *symbol = dlsym(RTLD_DEFAULT, name);
    require(symbol && dladdr(symbol, &info) && info.dli_fname, "symbol provider");
    require(std::filesystem::equivalent(info.dli_fname, expected), name);
}

struct Case {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void *owner = nullptr;
    ~Case() {
        xeon_iq3_cache_free_v1(owner);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

// Return the measured physical counts after checking resident pages and the full
// effective VMA policy. An allocation can share a VMA with another mapping.
static std::array<size_t, 2> check_pages(const void *base, size_t size) {
    constexpr size_t page = 4096;
    uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    require(sysconf(_SC_PAGESIZE) == page && begin % page == 0 && size % page == 0, "page geometry");
    std::ifstream policies("/proc/self/numa_maps"), maps("/proc/self/maps");
    require(policies.good() && maps.good(), "proc NUMA maps");
    std::map<uintptr_t, std::string> by_start;
    std::string line;
    while (std::getline(policies, line)) {
        std::istringstream fields(line);
        uintptr_t start;
        std::string policy;
        if (fields >> std::hex >> start >> policy) by_start.emplace(start, policy);
    }
    size_t covered = 0;
    while (std::getline(maps, line)) {
        unsigned long lo, hi;
        require(std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2, "VMA range");
        if (hi <= begin || lo >= begin + size) continue;
        require(by_start.at(lo) == "interleave:0-1", "compact VMA effective policy");
        covered += std::min<uintptr_t>(hi, begin + size) - std::max<uintptr_t>(lo, begin);
    }
    require(covered == size, "compact VMA coverage");
    std::vector<void *> addresses(size / page);
    std::vector<int> nodes(addresses.size(), -9999);
    for (size_t i = 0; i < addresses.size(); ++i) {
        addresses[i] = reinterpret_cast<void *>(begin + i * page);
    }
    if (syscall(SYS_move_pages, 0, addresses.size(), addresses.data(), nullptr, nodes.data(), 0) != 0) {
        throw std::runtime_error(std::string("move_pages query: ") + std::strerror(errno));
    }
    std::array<size_t, 2> counts{};
    for (size_t i = 0; i < nodes.size(); ++i) {
        require(nodes[i] == 0 || nodes[i] == 1, "compact page residency");
        require(nodes[i] == (nodes.front() ^ int(i % 2)), "compact pages interleaved across both sockets");
        ++counts[nodes[i]];
    }
    return counts;
}

static void check_unmapped(uintptr_t address) {
    unsigned char status = 0;
    errno = 0;
    require(mincore(reinterpret_cast<void *>(address), 4096, &status) == -1 && errno == ENOMEM,
            "released compact mapping");
}

static void run(ggml_backend_t cpu, ggml_type type, FILE *rows) {
    constexpr int k = 5120, m = 2304, n = 3;
    constexpr size_t packed_per_expert = size_t(k / 256) * m * 144;
    constexpr size_t packed_bytes = 2 * packed_per_expert;
    constexpr uint32_t selected[2] = {2, 0};
    constexpr int32_t route[n] = {0, 1, 2};
    Case c;
    c.ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(c.ctx, "ggml context");
    auto *w = ggml_new_tensor_3d(c.ctx, type, k, m, n);
    ggml_set_name(w, "blk.0.ffn_gate_exps.weight");
    auto *x = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, k, 1, 1);
    auto *ids = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, n, 1);
    auto *y = ggml_mul_mat_id(c.ctx, w, x, ids);
    auto *graph = ggml_new_graph_custom(c.ctx, 32, false);
    ggml_build_forward_expand(graph, y);
    c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, cpu);
    require(c.buffer, "CPU tensor buffer");
    ggml_backend_buffer_set_usage(c.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(8141);
    std::vector<unsigned char> source(ggml_nbytes(w));
    for (auto &v : source) v = static_cast<unsigned char>(rng());
    for (size_t off = 0; off < source.size(); off += ggml_type_size(type)) {
        uint16_t d = 0x1800 + 16 * ((off / ggml_type_size(type)) % 8);
        std::memcpy(source.data() + off, &d, sizeof(d));
    }
    auto set_weights = [&] { ggml_backend_tensor_set(w, source.data(), 0, source.size()); };
    set_weights();
    std::vector<float> activations(k);
    for (auto &v : activations) v = float(int(rng() % 2001) - 1000) / 2048;
    ggml_backend_tensor_set(x, activations.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids, route, 0, sizeof(route));
    auto compute = [&] {
        require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "graph compute");
        std::vector<float> values(m * n);
        require(ggml_nbytes(y) == values.size() * sizeof(float), "output size");
        ggml_backend_tensor_get(y, values.data(), 0, ggml_nbytes(y));
        require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), "finite output");
        require(std::fwrite(values.data(), sizeof(float), values.size(), rows) == values.size(), "save output");
        return values;
    };
    require(xeon_iq3_cache_total_v1() == 0, "empty registry at start");
    const auto reference = compute();
    for (int expert : route) {
        auto first = reference.begin() + expert * m;
        require(std::any_of(first, first + m, [](float v) { return v != 0; }), "nonzero reference expert");
    }
    c.owner = xeon_iq3_cache_create_v1(packed_bytes, size_t(64) << 30, 12);
    require(c.owner && xeon_iq3_cache_add_selected_v2(c.owner, w, selected, 2) == 1, "selected registration");
    require(xeon_iq3_cache_add_selected_v2(c.owner, w, selected, 2) == 0, "duplicate registration");
    require(xeon_iq3_cache_used_v1(c.owner) == packed_bytes && xeon_iq3_cache_total_v1() == packed_bytes, "accounting");
    const void *const *table = xeon_iq3_cache_lookup_experts_v2(w);
    require(table && table[0] && !table[1] && table[2] && !xeon_iq3_cache_lookup(w), "sparse membership");
    const uintptr_t address = reinterpret_cast<uintptr_t>(table[2]);
    require(reinterpret_cast<uintptr_t>(table[0]) == address + packed_per_expert, "sparse slot ordering");
    const auto nodes = check_pages(table[2], packed_bytes);
    const auto mixed = compute();
    require(!std::memcmp(reference.data(), mixed.data(), ggml_nbytes(y)), "mixed compact/native equality");

    // Clear the source scales for EVERY expert while leaving both cached copies intact.
    // The selected outputs must retain their original bytes and the single miss must
    // now use zero weights. A table hit counter alone cannot distinguish these paths.
    for (size_t off = 0; off < source.size(); off += ggml_type_size(type)) {
        source[off] = 0;
        source[off + 1] = 0;
    }
    set_weights();
    const auto changed = compute();
    for (uint32_t expert : selected) {
        require(!std::memcmp(reference.data() + expert * m, changed.data() + expert * m, m * sizeof(float)),
                "selected graph output uses retained compact weights");
    }
    require(std::all_of(changed.begin() + m, changed.begin() + 2 * m, [](float v) { return v == 0; }),
            "unselected graph output reads changed native weights");
    ggml_backend_synchronize(cpu);
    xeon_iq3_cache_free_v1(c.owner);
    c.owner = nullptr;
    check_unmapped(address);
    check_unmapped(address + packed_bytes - 4096);
    require(!xeon_iq3_cache_lookup_experts_v2(w) && xeon_iq3_cache_total_v1() == 0, "registry after release");
    const auto native = compute();
    require(std::all_of(native.begin(), native.end(), [](float v) { return v == 0; }), "all-native after release");
    c.owner = xeon_iq3_cache_create_v1(packed_bytes, size_t(64) << 30, 12);
    require(c.owner && xeon_iq3_cache_add_selected_v2(c.owner, w, selected, 2) == 1, "same-source reload");
    const auto reloaded = compute();
    require(std::all_of(reloaded.begin(), reloaded.end(), [](float v) { return v == 0; }), "fresh packed reload");
    std::printf("{\"type\":\"%s\",\"k\":%d,\"rows\":%d,\"expert_count\":%d,\"selected\":[2,0],"
                "\"routes\":[0,1,2],\"compact_bytes\":%zu,\"resident_pages_by_node\":[%zu,%zu],"
                "\"graphs\":5,\"confirmed_compact_slots\":2,\"confirmed_native_slots\":1}\n",
                ggml_type_name(type), k, m, n, packed_bytes, nodes[0], nodes[1]);
}

int main(int argc, char **argv) try {
    require(argc == 4, "usage: compact_quality expected-CPU-DSO expected-base-DSO fresh-output.f32");
    require(!std::getenv("GGML_CPU_IQ_ROW_LOCAL") && !std::getenv("GGML_CPU_IQ_AUDIT_DIR"),
            "ordinary profile has both row-local environment variables absent");
    require(prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0) == 1, "ordinary per-process THP setting");
    for (const char *name : {"ggml_backend_cpu_init", "xeon_iq3_cache_create_v1", "xeon_iq3_cache_add_selected_v2",
                             "xeon_iq3_cache_lookup_experts_v2", "xeon_iq3_cache_free_v1", "xeon_iq3_cache_total_v1"}) {
        check_provider(name, argv[1]);
    }
    check_provider("ggml_mul_mat_id", argv[2]);
    auto cpu = ggml_backend_cpu_init();
    require(cpu, "CPU backend");
    ggml_backend_cpu_set_n_threads(cpu, 12);
    FILE *rows = std::fopen(argv[3], "wbx");
    require(rows, "fresh raw F32 output");
    for (auto type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS}) run(cpu, type, rows);
    require(std::fclose(rows) == 0 && !xeon_iq3_cache_total_v1(), "final output and registry");
    ggml_backend_free(cpu);
    std::puts("{\"pass\":true,\"types\":3,\"graphs\":15,\"finite_f32_saved\":103680}");
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
}

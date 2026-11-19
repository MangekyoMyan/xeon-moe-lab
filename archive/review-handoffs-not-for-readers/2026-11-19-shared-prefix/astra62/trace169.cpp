#include "ggml-backend.h"
#include <cupti.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// One caller/context on the fixed teacher-forced diagnostic. The model and four
// ggml providers are unmodified. Only the two public ggml calls are interposed.
#define EXPORT extern "C" __attribute__((visibility("default")))

namespace {
constexpr uint64_t call_tag = UINT64_C(1) << 32;
constexpr uint64_t copy_tag = UINT64_C(2) << 32;
constexpr size_t buffer_bytes = 2 * 1024 * 1024;
constexpr size_t max_buffers = 24;
constexpr size_t max_calls = 8;
constexpr size_t max_splits = 1024;
constexpr size_t max_nodes = 65536;
constexpr size_t max_tensors = 81920;
constexpr size_t max_copies = 8192;
constexpr size_t table_size = 16384; // Per-call descriptor identities.
constexpr auto external_kind = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

struct Buffer {
    alignas(64) unsigned char bytes[buffer_bytes];
    enum { free, cupti, delivered } state = free;
    size_t valid = 0;
};
struct Call {
    unsigned case_id, call_id;
    uint64_t begin, end;
    size_t first_split, end_split;
};
struct Split {
    unsigned call, ordinal;
    char backend[32];
    int n_nodes, status;
    uint64_t metadata_begin, begin, end;
};
struct Tensor {
    uintptr_t self, data, buffer, view;
    uint64_t offset;
    int op, type, flags;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    char name[GGML_MAX_NAME];
    unsigned char params[GGML_MAX_OP_PARAMS];
};
struct Node {
    unsigned split, ordinal, tensor, src[GGML_MAX_SRC];
};
struct Copy {
    unsigned call, before_split, src, dst;
    uint64_t begin, end;
};
struct Kernel {
    uint64_t start, end, node;
    uint32_t context, device, stream, correlation, graph, name;
    int grid[3], block[3];
};
struct DeviceCopy {
    uint64_t start, end, bytes, node;
    uint32_t direction, src, dst, context, device, stream, correlation, runtime_correlation, graph;
};
struct ActivityApi {
    uint64_t start, end;
    uint32_t kind, cbid, pid, tid, correlation, result;
};
struct Correlation {
    uint64_t external;
    uint32_t correlation, kind;
};

std::array<Buffer, max_buffers> buffers;
std::mutex buffer_lock;
bool retain = false; // Protected by buffer_lock.
struct DropQueue { CUcontext context; uint32_t stream; };
std::array<DropQueue, 32> queues{};
size_t nqueues = 1; // Always check the global NULL/0 queue.
size_t requested = 0, released_outside = 0, delivered_inside = 0, highwater = 0;
std::atomic<unsigned> errors{0}, unknown_records{0}, foreign_thread_calls{0};
size_t dropped_inside = 0, dropped_outside = 0, unknown_kernel = 0;
uint32_t cupti_version = 0;
bool initialized = false, window_active = false, dumped = false;
std::atomic<bool> any_call{false};
thread_local int active_call = -1;
Call calls[max_calls];
Split splits[max_splits];
Tensor tensors[max_tensors];
Node nodes[max_nodes];
Copy copies[max_copies];
unsigned tensor_table[table_size];
size_t ncalls = 0, nsplits = 0, ntensors = 0, nnodes = 0, ncopies = 0;
std::vector<Kernel> kernels;
std::vector<DeviceCopy> device_copies;
std::vector<ActivityApi> apis;
std::vector<Correlation> correlations;
std::vector<std::string> names;
std::map<std::string, unsigned> name_ids;

void fail(const char * message, int code = 0) {
    if (errors.fetch_add(1) < 16) {
        std::fprintf(stderr, "trace169 error %s (%d)\n", message, code);
    }
}
bool checked(CUptiResult result, const char * operation) {
    if (result == CUPTI_SUCCESS) return true;
    const char * text = nullptr;
    cuptiGetResultString(result, &text);
    std::fprintf(stderr, "trace169 CUPTI %s: %s\n", operation, text ? text : "unknown");
    fail(operation, int(result));
    return false;
}
int status() { return errors.load() ? -1 : 0; }
template<class T> T symbol(const char * name) {
    void * p = dlsym(RTLD_NEXT, name);
    if (!p) {
        std::fprintf(stderr, "trace169 missing original ggml symbol %s\n", name);
        std::abort();
    }
    return reinterpret_cast<T>(p);
}
uint64_t now() {
    uint64_t value = 0;
    checked(cuptiGetTimestamp(&value), "timestamp");
    return value;
}
void push(uint64_t id) {
    checked(cuptiActivityPushExternalCorrelationId(external_kind, id), "push");
}
void pop(uint64_t expected) {
    uint64_t seen = 0;
    if (checked(cuptiActivityPopExternalCorrelationId(external_kind, &seen), "pop") && seen != expected) {
        fail("external stack");
    }
}
void CUPTIAPI request_buffer(uint8_t ** ptr, size_t * size, size_t * max_records) {
    std::lock_guard<std::mutex> hold(buffer_lock);
    *ptr = nullptr;
    *size = 0;
    *max_records = 0;
    size_t used = 0;
    Buffer * available = nullptr;
    for (auto & b : buffers) {
        if (b.state != Buffer::free) ++used;
        else if (!available) available = &b;
    }
    if (!available) {
        fail("buffer pool exhausted");
        return;
    }
    ++requested;
    highwater = std::max(highwater, used + 1);
    available->state = Buffer::cupti;
    *ptr = available->bytes;
    *size = buffer_bytes;
}
void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t * ptr, size_t size, size_t valid) {
    std::lock_guard<std::mutex> hold(buffer_lock);
    const auto end = queues.begin() + nqueues;
    if (std::find_if(queues.begin(), end, [&](const DropQueue & q) {
            return q.context == context && q.stream == stream;
        }) == end) {
        if (nqueues < queues.size()) queues[nqueues++] = {context, stream};
        else fail("drop queue capacity");
    }
    for (auto & b : buffers) {
        if (b.bytes == ptr && b.state == Buffer::cupti && valid <= size && size == buffer_bytes) {
            if (retain) {
                b.valid = valid;
                b.state = Buffer::delivered;
                ++delivered_inside;
            } else {
                b.state = Buffer::free;
                ++released_outside;
            }
            return;
        }
    }
    fail("unknown or invalid callback buffer");
}
unsigned intern_tensor(const ggml_tensor * t) {
    if (!t) return 0;
    uintptr_t key = reinterpret_cast<uintptr_t>(t);
    size_t slot = ((key >> 5) * UINT64_C(11400714819323198485)) >> 50;
    for (size_t n = 0; n < table_size; ++n) {
        unsigned id = tensor_table[slot];
        if (id && tensors[id - 1].self == key) return id;
        if (!id) {
            if (ntensors == max_tensors) {
                fail("tensor capacity");
                return 0;
            }
            Tensor & v = tensors[ntensors++];
            v.self = key;
            v.data = reinterpret_cast<uintptr_t>(t->data);
            v.buffer = reinterpret_cast<uintptr_t>(t->buffer);
            v.view = reinterpret_cast<uintptr_t>(t->view_src);
            v.offset = t->view_offs;
            v.op = t->op;
            v.type = t->type;
            v.flags = t->flags;
            std::memcpy(v.ne, t->ne, sizeof(v.ne));
            std::memcpy(v.nb, t->nb, sizeof(v.nb));
            std::memcpy(v.name, t->name, sizeof(v.name));
            v.name[sizeof(v.name) - 1] = '\0';
            std::memcpy(v.params, t->op_params, sizeof(v.params));
            tensor_table[slot] = unsigned(ntensors);
            return unsigned(ntensors);
        }
        slot = (slot + 1) & (table_size - 1);
    }
    fail("per-call tensor table");
    return 0;
}
void capture_graph(ggml_backend_t backend, ggml_cgraph * graph, Split & s, unsigned split_id) {
    static auto bname = symbol<decltype(&ggml_backend_name)>("ggml_backend_name");
    static auto count = symbol<decltype(&ggml_graph_n_nodes)>("ggml_graph_n_nodes");
    static auto entries = symbol<decltype(&ggml_graph_nodes)>("ggml_graph_nodes");
    const char * backend_name = bname(backend);
    if (!backend_name || std::strlen(backend_name) >= sizeof(s.backend)) {
        fail("backend name");
        return;
    }
    std::strcpy(s.backend, backend_name);
    s.n_nodes = count(graph);
    ggml_tensor ** graph_nodes = entries(graph);
    if (s.n_nodes <= 0 || size_t(s.n_nodes) > max_nodes - nnodes) {
        fail("node count", s.n_nodes);
        return;
    }
    for (int i = 0; i < s.n_nodes; ++i) {
        const ggml_tensor * t = graph_nodes[i];
        Node & n = nodes[nnodes++];
        n.split = split_id;
        n.ordinal = unsigned(i);
        n.tensor = intern_tensor(t);
        for (unsigned j = 0; j < GGML_MAX_SRC; ++j) n.src[j] = intern_tensor(t->src[j]);
    }
}
template<class T> bool has_room(const std::vector<T> & rows, size_t cap) {
    if (rows.size() < cap) return true;
    fail("activity capacity");
    return false;
}
unsigned kernel_name(const char * name) {
    if (!name || !*name || std::strlen(name) >= 4096) {
        fail("kernel name");
        return 0;
    }
    auto found = name_ids.find(name);
    if (found != name_ids.end()) return found->second;
    if (!has_room(names, 2048)) return 0;
    unsigned id = unsigned(names.size());
    names.emplace_back(name);
    name_ids.emplace(name, id);
    return id;
}
void collect(const CUpti_Activity * a) {
    switch (a->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto * k = reinterpret_cast<const CUpti_ActivityKernel9 *>(a);
            if (!has_room(kernels, 65536)) return;
            kernels.push_back({k->start, k->end, k->graphNodeId, k->contextId, k->deviceId, k->streamId,
                k->correlationId, k->graphId, kernel_name(k->name), {k->gridX, k->gridY, k->gridZ},
                {k->blockX, k->blockY, k->blockZ}});
            if (!k->start || k->end < k->start || !k->correlationId) fail("kernel timestamps or correlation");
            return;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto * m = reinterpret_cast<const CUpti_ActivityMemcpy6 *>(a);
            if (!has_room(device_copies, 32768)) return;
            device_copies.push_back({m->start, m->end, m->bytes, m->graphNodeId, m->copyKind, m->srcKind,
                m->dstKind, m->contextId, m->deviceId, m->streamId, m->correlationId,
                m->runtimeCorrelationId, m->graphId});
            if (!m->start || m->end < m->start) fail("memcpy timestamps");
            return;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API: {
            auto * v = reinterpret_cast<const CUpti_ActivityAPI *>(a);
            if (!has_room(apis, 262144)) return;
            apis.push_back({v->start, v->end, unsigned(a->kind), v->cbid, v->processId, v->threadId,
                v->correlationId, v->returnValue});
            return;
        }
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
            auto * x = reinterpret_cast<const CUpti_ActivityExternalCorrelation *>(a);
            if (!has_room(correlations, 262144)) return;
            correlations.push_back({x->externalId, x->correlationId, unsigned(x->externalKind)});
            if (x->externalKind != external_kind) fail("unexpected external kind", int(x->externalKind));
            return;
        }
        default:
            ++unknown_records;
            fail("unknown activity kind", int(a->kind));
    }
}
void flush(bool keep) {
    if (!checked(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush")) return;
    std::array<DropQueue, 32> snapshot;
    size_t count;
    {
        std::lock_guard<std::mutex> hold(buffer_lock);
        snapshot = queues;
        count = nqueues;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t dropped = 0;
        if (checked(cuptiActivityGetNumDroppedRecords(snapshot[i].context, snapshot[i].stream, &dropped), "dropped")) {
            (keep ? dropped_inside : dropped_outside) += dropped;
            if (keep && dropped) fail("window CUPTI dropped records", int(dropped));
        }
    }
    // Flush has delivered the completed buffers. The single caller is outside
    // model work here. No file I/O, parsing or name copying in CUPTI callbacks.
    if (!keep) return;
    for (auto & b : buffers) {
        {
            std::lock_guard<std::mutex> hold(buffer_lock);
            if (b.state != Buffer::delivered) continue;
        }
        CUpti_Activity * a = nullptr;
        while (b.valid) {
            CUptiResult result = cuptiActivityGetNextRecord(b.bytes, b.valid, &a);
            if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            if (!checked(result, "record")) break;
            collect(a);
        }
        std::lock_guard<std::mutex> hold(buffer_lock);
        b.state = Buffer::free;
    }
}
constexpr CUpti_ActivityKind window_kinds[] = {
    CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_MEMCPY
};
template<class F> void output(const char * dir, const char * suffix, F write) {
    std::string path = std::string(dir) + "/trace169-" + suffix;
    FILE * f = std::fopen(path.c_str(), "wx");
    if (!f) {
        std::perror(path.c_str());
        fail("open output");
        return;
    }
    write(f);
    bool bad = std::ferror(f);
    if (std::fclose(f) != 0 || bad) fail("write output");
}
} // namespace

EXPORT int s169_trace_initialize() {
    if (initialized || dumped) { fail("double initialize"); return -1; }
    // Establish graph-node tracking before the application's first CUDA graph
    // is instantiated. Outside trace windows the callbacks immediately recycle
    // every buffer. No driver/runtime/memcpy/external records are requested.
    if (!checked(cuptiGetVersion(&cupti_version), "version") ||
        !checked(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer), "callbacks") ||
        !checked(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable kernel")) return -1;
    initialized = true;
    return status();
}
EXPORT int s169_trace_window_begin() {
    if (!initialized || dumped || window_active || active_call >= 0) { fail("window begin state"); return -1; }
    flush(false); // preceding prefix/warmup output is already ready in caller
    for (auto kind : window_kinds) {
        if (!checked(cuptiActivityEnable(kind), "enable window activity")) return -1;
    }
    window_active = true;
    std::lock_guard<std::mutex> hold(buffer_lock);
    retain = true;
    return status();
}
EXPORT int s169_trace_call_begin(unsigned case_id, unsigned call_id) {
    if (!window_active || active_call >= 0 || ncalls == max_calls || case_id > 1 || call_id < 1 || call_id > 4) {
        fail("call begin state"); return -1;
    }
    for (size_t i = 0; i < ncalls; ++i) {
        if (calls[i].case_id == case_id && calls[i].call_id == call_id) { fail("duplicate case/call"); return -1; }
    }
    std::fill_n(tensor_table, table_size, 0);
    active_call = int(ncalls++);
    Call & c = calls[active_call];
    c.case_id = case_id;
    c.call_id = call_id;
    c.first_split = nsplits;
    push(call_tag + ncalls);
    c.begin = now();
    any_call.store(true, std::memory_order_release);
    return status();
}
EXPORT int s169_trace_call_end() {
    if (!window_active || active_call < 0) { fail("call end state"); return -1; }
    Call & c = calls[active_call];
    c.end = now();
    c.end_split = nsplits;
    any_call.store(false, std::memory_order_release);
    pop(call_tag + unsigned(active_call) + 1);
    active_call = -1;
    if (!c.begin || c.end <= c.begin || c.first_split == c.end_split) fail("empty call trace");
    return status();
}
EXPORT int s169_trace_window_end() {
    if (!window_active || active_call >= 0) { fail("window end state"); return -1; }
    flush(true); // preceding fourth existing output-ready boundary is in caller
    {
        std::lock_guard<std::mutex> hold(buffer_lock);
        retain = false;
    }
    for (auto it = std::rbegin(window_kinds); it != std::rend(window_kinds); ++it) {
        checked(cuptiActivityDisable(*it), "disable window activity");
    }
    window_active = false;
    return status();
}
EXPORT enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph * graph) {
    static auto original = symbol<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");
    if (active_call < 0) {
        if (any_call.load(std::memory_order_acquire)) ++foreign_thread_calls;
        return original(backend, graph);
    }
    if (nsplits == max_splits) { fail("split capacity"); return original(backend, graph); }
    unsigned id = unsigned(++nsplits);
    Split & s = splits[id - 1];
    s.call = unsigned(active_call) + 1;
    s.ordinal = unsigned((nsplits - 1) - calls[active_call].first_split);
    s.metadata_begin = now();
    capture_graph(backend, graph, s, id);
    push(id);
    s.begin = now();
    enum ggml_status result = original(backend, graph);
    s.end = now();
    pop(id);
    s.status = int(result);
    if (result != GGML_STATUS_SUCCESS) fail("backend status", int(result));
    return result;
}
EXPORT void ggml_backend_tensor_copy(const ggml_tensor * src, ggml_tensor * dst) {
    static auto original = symbol<decltype(&ggml_backend_tensor_copy)>("ggml_backend_tensor_copy");
    if (active_call < 0) {
        if (any_call.load(std::memory_order_acquire)) ++foreign_thread_calls;
        original(src, dst);
        return;
    }
    if (ncopies == max_copies) { fail("copy capacity"); original(src, dst); return; }
    uint64_t id = copy_tag + ++ncopies;
    Copy & c = copies[ncopies - 1];
    c.call = unsigned(active_call) + 1;
    c.before_split = unsigned(nsplits - calls[active_call].first_split);
    c.src = intern_tensor(src);
    c.dst = intern_tensor(dst);
    push(id);
    c.begin = now();
    original(src, dst);
    c.end = now();
    pop(id);
}
EXPORT int s169_trace_dump(const char * dir) {
    if (!initialized || !dir || !*dir || window_active || active_call >= 0 || dumped) {
        fail("dump state"); return -1;
    }
    flush(false);
    checked(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "disable kernel");
    dumped = true;
    std::map<unsigned, uint64_t> correlation_map;
    for (const auto & c : correlations) {
        auto [it, fresh] = correlation_map.emplace(c.correlation, c.external);
        if (!fresh && it->second != c.external) fail("conflicting external correlation");
        bool known = (c.external > 0 && c.external <= nsplits) ||
            (c.external > call_tag && c.external <= call_tag + ncalls) ||
            (c.external > copy_tag && c.external <= copy_tag + ncopies);
        if (!known) fail("unknown external ID");
    }
    for (const auto & k : kernels) {
        if (!correlation_map.count(k.correlation)) ++unknown_kernel;
    }
    if (!ncalls || !nsplits || !ncopies || kernels.empty() || apis.empty() || correlations.empty() ||
        device_copies.empty() || unknown_kernel || foreign_thread_calls.load()) fail("incomplete scoped trace");
    output(dir, "providers.tsv", [](FILE * f) {
        std::fputs("symbol\tpath\tfile_device\tinode\tmapped_base_hex\tfunction_hex\n", f);
        for (const char * name : {"ggml_backend_graph_compute_async", "ggml_backend_tensor_copy", "cuptiGetTimestamp"}) {
            void * p = symbol<void *>(name);
            Dl_info info{};
            char * resolved = nullptr;
            struct stat st{};
            if (!dladdr(p, &info) || !info.dli_fname || !(resolved = realpath(info.dli_fname, nullptr)) ||
                stat(resolved, &st) != 0) {
                fail("original provider identity");
            } else {
                std::fprintf(f, "%s\t%s\t%" PRIuMAX "\t%" PRIuMAX "\t%" PRIxPTR "\t%" PRIxPTR "\n",
                    name, resolved, uintmax_t(st.st_dev), uintmax_t(st.st_ino),
                    reinterpret_cast<uintptr_t>(info.dli_fbase), reinterpret_cast<uintptr_t>(p));
            }
            std::free(resolved);
        }
    });
    output(dir, "calls.tsv", [](FILE * f) {
        std::fputs("call\tcase_id\tcall_id\tstart_ns\tend_ns\tn_splits\n", f);
        for (size_t i = 0; i < ncalls; ++i) {
            const auto & c = calls[i];
            std::fprintf(f, "%zu\t%u\t%u\t%" PRIu64 "\t%" PRIu64 "\t%zu\n", i+1, c.case_id, c.call_id,
                c.begin, c.end, c.end_split - c.first_split);
        }
    });
    output(dir, "splits.tsv", [](FILE * f) {
        std::fputs("split\tcall\tordinal\tbackend\tn_nodes\tmetadata_start_ns\tdispatch_ns\treturn_ns\tstatus\n", f);
        for (size_t i = 0; i < nsplits; ++i) {
            const auto & s = splits[i];
            std::fprintf(f, "%zu\t%u\t%u\t%s\t%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%d\n",
                i+1, s.call, s.ordinal, s.backend, s.n_nodes, s.metadata_begin, s.begin, s.end, s.status);
        }
    });
    output(dir, "nodes.tsv", [](FILE * f) {
        std::fputs("split\tordinal\ttensor", f);
        for (unsigned j = 0; j < GGML_MAX_SRC; ++j) std::fprintf(f, "\tsrc%u", j);
        std::fputc('\n', f);
        for (size_t i = 0; i < nnodes; ++i) {
            const auto & n = nodes[i];
            std::fprintf(f, "%u\t%u\t%u", n.split, n.ordinal, n.tensor);
            for (auto src : n.src) std::fprintf(f, "\t%u", src);
            std::fputc('\n', f);
        }
    });
    output(dir, "tensors.tsv", [](FILE * f) {
        auto op_name = symbol<decltype(&ggml_op_name)>("ggml_op_name");
        auto type_name = symbol<decltype(&ggml_type_name)>("ggml_type_name");
        std::fputs("tensor\tname\top\ttype\tflags\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tself_hex\tdata_hex\tbuffer_hex\tview_hex\tview_offset\tparams_hex\n", f);
        for (size_t i = 0; i < ntensors; ++i) {
            const auto & t = tensors[i];
            std::fprintf(f, "%zu\t%s\t%s\t%s\t%d", i+1, t.name, op_name(ggml_op(t.op)), type_name(ggml_type(t.type)), t.flags);
            for (auto n : t.ne) std::fprintf(f, "\t%" PRId64, n);
            for (auto n : t.nb) std::fprintf(f, "\t%zu", n);
            std::fprintf(f, "\t%" PRIxPTR "\t%" PRIxPTR "\t%" PRIxPTR "\t%" PRIxPTR "\t%" PRIu64 "\t",
                t.self, t.data, t.buffer, t.view, t.offset);
            for (auto b : t.params) std::fprintf(f, "%02x", unsigned(b));
            std::fputc('\n', f);
        }
    });
    output(dir, "copies.tsv", [](FILE * f) {
        std::fputs("copy\tcall\tbefore_split_ordinal\tsrc\tdst\tstart_ns\treturn_ns\n", f);
        for (size_t i = 0; i < ncopies; ++i) {
            const auto & c = copies[i];
            std::fprintf(f, "%zu\t%u\t%u\t%u\t%u\t%" PRIu64 "\t%" PRIu64 "\n", i+1, c.call,
                c.before_split, c.src, c.dst, c.begin, c.end);
        }
    });
    output(dir, "kernels.tsv", [](FILE * f) {
        std::fputs("start_ns\tend_ns\tcontext\tdevice\tstream\tcorrelation\tgraph\tgraph_node\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_y\tblock_z\tname\n", f);
        for (const auto & k : kernels) {
            std::fprintf(f, "%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%u\t%u\t%u\t%" PRIu64 "\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n",
                k.start, k.end, k.context, k.device, k.stream, k.correlation, k.graph, k.node,
                k.grid[0], k.grid[1], k.grid[2], k.block[0], k.block[1], k.block[2],
                k.name < names.size() ? names[k.name].c_str() : "");
        }
    });
    output(dir, "device-copies.tsv", [](FILE * f) {
        std::fputs("start_ns\tend_ns\tbytes\tdirection\tsrc_kind\tdst_kind\tcontext\tdevice\tstream\tcorrelation\truntime_correlation\tgraph\tgraph_node\n", f);
        for (const auto & m : device_copies) {
            std::fprintf(f, "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%" PRIu64 "\n",
                m.start, m.end, m.bytes, m.direction, m.src, m.dst, m.context, m.device, m.stream,
                m.correlation, m.runtime_correlation, m.graph, m.node);
        }
    });
    output(dir, "api.tsv", [](FILE * f) {
        std::fputs("start_ns\tend_ns\tkind\tcbid\tpid\ttid\tcorrelation\tresult\n", f);
        for (const auto & a : apis) {
            std::fprintf(f, "%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%u\t%u\t%u\t%u\n",
                a.start, a.end, a.kind, a.cbid, a.pid, a.tid, a.correlation, a.result);
        }
    });
    output(dir, "correlations.tsv", [](FILE * f) {
        std::fputs("correlation\texternal_id\texternal_kind\n", f);
        for (const auto & c : correlations) std::fprintf(f, "%u\t%" PRIu64 "\t%u\n", c.correlation, c.external, c.kind);
    });
    output(dir, "summary.json", [](FILE * f) {
        std::lock_guard<std::mutex> hold(buffer_lock);
        std::fprintf(f,
            "{\"schema\":1,\"cupti_version\":%u,\"calls\":%zu,\"splits\":%zu,\"nodes\":%zu,\"tensors\":%zu,"
            "\"host_copies\":%zu,\"kernels\":%zu,\"api\":%zu,\"device_copies\":%zu,\"correlations\":%zu,"
            "\"unknown_kernel_correlation\":%zu,\"foreign_thread_calls\":%u,\"dropped_inside\":%zu,"
            "\"dropped_outside\":%zu,\"unknown_records\":%u,\"errors\":%u,\"pool_highwater\":%zu,"
            "\"drop_queues\":%zu,\"pool_capacity\":%zu,\"buffer_bytes\":%zu,\"buffer_requests\":%zu,\"buffers_delivered_inside\":%zu,"
            "\"buffers_released_outside\":%zu}\n",
            cupti_version, ncalls, nsplits, nnodes, ntensors, ncopies, kernels.size(), apis.size(),
            device_copies.size(), correlations.size(), unknown_kernel, foreign_thread_calls.load(), dropped_inside,
            dropped_outside, unknown_records.load(), errors.load(), highwater, nqueues, max_buffers, buffer_bytes,
            requested, delivered_inside, released_outside);
    });
    return status();
}

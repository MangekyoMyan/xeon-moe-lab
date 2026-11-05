#include "ggml-cpu.h"
#include "numa-replica.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ops.h"
#include "traits.h"

#if defined(__linux__)
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
struct replica_context;

class replica_traits final : public ggml::cpu::tensor_traits {
    replica_context * context;
  public:
    explicit replica_traits(replica_context * context) : context(context) {}
    bool work_size(int, const ggml_tensor *, size_t &) override { return false; }
    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override;
};

struct replica_context {
    void * data[2] = {nullptr, nullptr};
    size_t size;
    bool trace;
    std::atomic<uint64_t> calls[3]{};
    replica_traits traits;
    explicit replica_context(size_t size) : size(size), traits(this) {
        const char * env = std::getenv("GGML_CPU_NUMA_REPLICA_TRACE");
        trace = env && std::strcmp(env, "1") == 0;
    }
};

static bool eligible(const ggml_tensor * weight) {
    if (weight->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(weight) || ggml_n_dims(weight) != 2) {
        return false;
    }
    return (weight->ne[0] == 1024 && (weight->ne[1] == 32768 || weight->ne[1] == 8192)) ||
           (weight->ne[0] == 8192 && weight->ne[1] == 4096);
}

static size_t tensor_offset(const replica_context * context, const ggml_tensor * tensor, size_t offset, size_t size) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(context->data[0]);
    const uintptr_t ptr = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(ptr >= base && ptr - base <= context->size);
    const size_t start = ptr - base;
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    GGML_ASSERT(offset <= context->size - start && size <= context->size - start - offset);
    return start + offset;
}

bool replica_traits::compute_forward(ggml_compute_params * params, ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || op->src[0]->type != GGML_TYPE_Q8_0) {
        return false;
    }
    unsigned node = 2;
    if (syscall(SYS_getcpu, nullptr, &node, nullptr) != 0 || node > 1) node = 2;
    if (context->trace) context->calls[node].fetch_add(1, std::memory_order_relaxed);
    const size_t offset = tensor_offset(context, op->src[0], 0, ggml_nbytes(op->src[0]));
    ggml_tensor weight = *op->src[0];
    weight.data = static_cast<char *>(context->data[node < 2 ? node : 0]) + offset;
    ggml_tensor local_op = *op;
    local_op.src[0] = &weight;
    ggml_compute_forward_mul_mat(params, &local_op);
    return true;
}

static void release(replica_context * context) {
    for (int node = 0; node < 2; ++node) {
        if (context->data[node]) munmap(context->data[node], context->size);
    }
    delete context;
}

static void buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<replica_context *>(buffer->context);
    if (context->trace && context->size) {
        std::fprintf(stderr, "NUMA_REPLICA_CALLS node0=%llu node1=%llu fallback=%llu\n",
            static_cast<unsigned long long>(context->calls[0].load()),
            static_cast<unsigned long long>(context->calls[1].load()),
            static_cast<unsigned long long>(context->calls[2].load()));
    }
    release(context);
}

static void * buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<replica_context *>(buffer->context)->data[0];
}

static ggml_status buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * context = static_cast<replica_context *>(buffer->context);
    tensor->extra = &context->traits;
    if (context->trace) {
        std::fprintf(stderr, "NUMA_REPLICA_TENSOR %s bytes=%zu offset=%zu\n", tensor->name,
            ggml_nbytes(tensor), tensor_offset(context, tensor, 0, ggml_nbytes(tensor)));
    }
    return GGML_STATUS_SUCCESS;
}

static void buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = static_cast<replica_context *>(buffer->context);
    const size_t start = tensor_offset(context, tensor, offset, size);
    for (int node = 0; node < 2; ++node) {
        std::memcpy(static_cast<char *>(context->data[node]) + start, data, size);
    }
}

static void buffer_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = static_cast<replica_context *>(buffer->context);
    const size_t start = tensor_offset(context, tensor, offset, size);
    std::memcpy(data, static_cast<char *>(context->data[0]) + start, size);
}

static void buffer_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = static_cast<replica_context *>(buffer->context);
    const size_t start = tensor_offset(context, tensor, offset, size);
    for (int node = 0; node < 2; ++node) std::memset(static_cast<char *>(context->data[node]) + start, value, size);
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<replica_context *>(buffer->context);
    for (int node = 0; node < 2; ++node) {
        if (context->size) std::memset(context->data[node], value, context->size);
    }
}

static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // Bound this private experiment to the profiled 3 GiB weight set.
    if (size > size_t(4) * 1024 * 1024 * 1024) return nullptr;
    auto * context = new (std::nothrow) replica_context(size);
    if (!context) return nullptr;
    for (int node = 0; size && node < 2; ++node) {
        void * data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data == MAP_FAILED) {
            GGML_LOG_ERROR("NUMA_REPLICA mmap failed: %s\n", std::strerror(errno));
            release(context);
            return nullptr;
        }
        context->data[node] = data;
        const unsigned long mask = 1UL << node;
        if (syscall(SYS_mbind, data, size, MPOL_BIND, &mask, sizeof(mask) * 8, 0) != 0) {
            GGML_LOG_ERROR("NUMA_REPLICA mbind node %d failed: %s\n", node, std::strerror(errno));
            release(context);
            return nullptr;
        }
        if (context->trace) std::fprintf(stderr, "NUMA_REPLICA_MAP node=%d address=%p bytes=%zu\n", node, data, size);
    }
    ggml_backend_buffer_i iface{};
    iface.free_buffer = buffer_free;
    iface.get_base = buffer_base;
    iface.init_tensor = buffer_init_tensor;
    iface.memset_tensor = buffer_memset;
    iface.set_tensor = buffer_set;
    iface.get_tensor = buffer_get;
    iface.clear = buffer_clear;
    return ggml_backend_buffer_init(buft, iface, context, size);
}

static const char * buffer_name(ggml_backend_buffer_type_t) { return "CPU_NUMA_REPLICA"; }
static size_t buffer_alignment(ggml_backend_buffer_type_t) { return TENSOR_ALIGNMENT; }

class replica_buffer_type final : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        return op->op == GGML_OP_MUL_MAT && eligible(op->src[0]) &&
            op->src[0]->buffer && op->src[0]->buffer->buft->context == this &&
            op->src[1]->type == GGML_TYPE_F32 &&
            (!op->src[1]->buffer || ggml_backend_buffer_is_host(op->src[1]->buffer));
    }
    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft->context == this) {
            return &static_cast<replica_context *>(op->src[0]->buffer->context)->traits;
        }
        return nullptr;
    }
};
} // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_numa_replica_buffer_type() {
    const char * env = std::getenv("GGML_CPU_NUMA_REPLICA");
    if (!env || std::strcmp(env, "1") != 0) return nullptr;
    static replica_buffer_type extra;
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ {buffer_name, alloc_buffer, nullptr, buffer_alignment,
                          nullptr, nullptr, nullptr, nullptr},
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &extra,
    };
    return &buft;
}
#else
ggml_backend_buffer_type_t ggml_backend_cpu_numa_replica_buffer_type() { return nullptr; }
#endif

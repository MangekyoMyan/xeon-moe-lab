// Included in the isolated copy of gpu-stream-20261103/ggml-backend.cpp.
// T=1 CPU-routed DSV4, pinned CUDA45/CPU33a, one scheduler caller.
#include <array>
#include <new>

struct prefix169_pair {
    int cpu_split = -1;
    int gpu_split = -1;
    ggml_backend_t device = nullptr;
    ggml_cgraph early{};
    ggml_cgraph tail{};
    std::vector<ggml_tensor *> tail_nodes;
    // Original GPU input and output bindings must still be those seen when
    // assigning the view UIDs. Shape/strides and nodes are generation-bound.
    std::array<ggml_tensor *, 9> checked{};
    std::array<void *, 9> expected_data{};
    std::array<ggml_backend_buffer_t, 9> expected_buffer{};
};
struct prefix169_state {
    static constexpr unsigned n_layers = 37;
    std::array<prefix169_pair, n_layers> pairs;
    std::vector<prefix169_pair *> early_at, tail_at;
    ggml_context * scratch_ctx = nullptr;
    ggml_backend_buffer_t scratch_buffer = nullptr;
    ggml_backend_t device = nullptr;
    ggml_tensor * scratch[3]{};
    uint64_t last_graph_uid = 0, plans = 0, early_submits = 0, tail_submits = 0, fault_drains = 0;
    bool enabled_graph = false;
};
static prefix169_state * prefix169_new() {
    const char * enable = getenv("XEON_169_SHARED_PREFIX");
    const char * multi = getenv("GGML_CUDA_GRAPH_OPT");
    if (!enable || strcmp(enable, "1") != 0 || (multi && atoi(multi) != 0)) {
        return nullptr;
    }
    return new (std::nothrow) prefix169_state;
}
static void prefix169_free(prefix169_state * state) {
    if (!state) return;
    // Scratches are shared across sequential layers and graphs. Release only
    // after the last possible device access, including on context teardown.
    if (state->device && state->early_submits) ggml_backend_synchronize(state->device);
    ggml_backend_buffer_free(state->scratch_buffer);
    ggml_free(state->scratch_ctx);
    delete state;
}
static bool prefix169_named(const ggml_tensor * t, const char * stem, int layer) {
    char label[GGML_MAX_NAME];
    snprintf(label, sizeof(label), "%s-%d", stem, layer);
    return t && strcmp(t->name, label) == 0;
}
static bool prefix169_weight(const ggml_tensor * t, const char * part, int layer) {
    char label[GGML_MAX_NAME];
    snprintf(label, sizeof(label), "blk.%d.ffn_%s_shexp.weight", layer, part);
    return t && strcmp(t->name, label) == 0 && t->buffer && t->data &&
           ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}
static bool prefix169_shape(const ggml_tensor * t, ggml_type type, int64_t a, int64_t b) {
    return t && t->data && t->buffer && t->type == type && t->ne[0] == a && t->ne[1] == b &&
           t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
}
static bool prefix169_same_device(ggml_backend_t cuda, const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_get_type(t->buffer) == ggml_backend_get_default_buffer_type(cuda);
}
static bool prefix169_scratch(prefix169_state & s, ggml_backend_t device) {
    if (s.scratch_buffer) return s.device == device;
    // Every slot has a separate buffer address; this also covers allocators
    // that normally reuse gate/up storage in place for GLU.
    const size_t overhead = 3 * ggml_tensor_overhead() + 4 * GGML_MEM_ALIGN;
    s.scratch_ctx = ggml_init({overhead, nullptr, true});
    if (!s.scratch_ctx) return false;
    for (ggml_tensor * & t : s.scratch) {
        t = ggml_new_tensor_1d(s.scratch_ctx, GGML_TYPE_F32, 2304);
    }
    s.scratch_buffer = ggml_backend_alloc_ctx_tensors(s.scratch_ctx, device);
    if (!s.scratch_buffer) {
        ggml_free(s.scratch_ctx);
        s.scratch_ctx = nullptr;
        return false;
    }
    s.device = device;
    ggml_backend_buffer_set_usage(s.scratch_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    for (int i = 0; i < 3; ++i) {
        GGML_ASSERT(prefix169_shape(s.scratch[i], GGML_TYPE_F32, 2304, 1));
        const uintptr_t a = reinterpret_cast<uintptr_t>(s.scratch[i]->data);
        for (int j = 0; j < i; ++j) {
            const uintptr_t b = reinterpret_cast<uintptr_t>(s.scratch[j]->data);
            GGML_ASSERT(a + 2304*sizeof(float) <= b || b + 2304*sizeof(float) <= a);
        }
    }
    return true;
}
static bool prefix169_pair_prepare(ggml_backend_sched_t sched, int ci, int layer, prefix169_pair & p) {
    auto & cpu = sched->splits[ci];
    auto & gpu = sched->splits[ci+1];
    ggml_backend_t host = sched->backends[cpu.backend_id], device = sched->backends[gpu.backend_id];
    if (strcmp(ggml_backend_name(host), "CPU") != 0 || strcmp(ggml_backend_name(device), "CUDA0") != 0 ||
        cpu.n_inputs < 2 || gpu.n_inputs < 1 || gpu.graph.n_nodes < 7) return false;
    ggml_tensor * routed[4]{};
    int k = 0;
    for (int j = 0; j < cpu.graph.n_nodes; ++j) {
        ggml_tensor * n = cpu.graph.nodes[j];
        if (ggml_is_view_op(n->op)) continue;
        if (k == 4 || !n->buffer || !ggml_backend_buffer_is_host(n->buffer)) return false;
        for (const auto * src : n->src) {
            if (src && (!src->buffer || !ggml_backend_buffer_is_host(src->buffer))) return false;
        }
        routed[k++] = n;
    }
    if (k != 4) return false;
    const char * routed_names[] = {"ffn_moe_gate","ffn_moe_up","ffn_moe_swiglu_limited","ffn_moe_down"};
    for (int j = 0; j < 4; ++j) {
        if (!prefix169_named(routed[j], routed_names[j], layer) ||
            routed[j]->op != (j == 2 ? GGML_OP_GLU : GGML_OP_MUL_MAT_ID) ||
            !prefix169_shape(routed[j], GGML_TYPE_F32, j == 3 ? 5120 : 2304, 6)) return false;
    }
    ggml_tensor * x = routed[0]->src[1], * ids = routed[0]->src[2];
    if (!prefix169_shape(x, GGML_TYPE_F32, 5120, 1) || !prefix169_shape(ids, GGML_TYPE_I32, 6, 1) ||
        routed[1]->src[1] != x || routed[1]->src[2] != ids || routed[3]->src[2] != ids ||
        routed[2]->src[0] != routed[0] || routed[2]->src[1] != routed[1] || routed[3]->src[1] != routed[2]) return false;
    ggml_tensor * norm_source = nullptr;
    bool copied_ids = false;
    for (int j = 0; j < cpu.n_inputs; ++j) {
        ggml_tensor * input = cpu.inputs[j];
        ggml_tensor * cp = tensor_copy(input, cpu.backend_id, sched->cur_copy);
        if (cp == x) norm_source = input;
        if (cp == ids) copied_ids = true;
    }
    ggml_tensor * weighted = gpu.graph.nodes[0];
    if (!norm_source || !copied_ids || weighted->op != GGML_OP_MUL ||
        !prefix169_named(weighted, "ffn_moe_weighted", layer) ||
        weighted->src[0] != tensor_copy(routed[3], gpu.backend_id, sched->cur_copy) ||
        std::find(gpu.inputs,gpu.inputs+gpu.n_inputs,routed[3]) == gpu.inputs+gpu.n_inputs) return false;
    int at = 0;
    while (at + 4 < gpu.graph.n_nodes && !prefix169_named(gpu.graph.nodes[at], "ffn_gate", layer)) ++at;
    if (at <= 0 || at + 4 >= gpu.graph.n_nodes) return false;
    ggml_tensor ** a = gpu.graph.nodes + at;
    const char * shared_names[] = {"ffn_gate","ffn_up","ffn_swiglu_limited","ffn_shexp","ffn_out"};
    const ggml_op ops[] = {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT, GGML_OP_ADD};
    for (int j = 0; j < 5; ++j) {
        if (!prefix169_named(a[j], shared_names[j], layer) || a[j]->op != ops[j] ||
            !prefix169_same_device(device, a[j]) || !prefix169_shape(a[j], GGML_TYPE_F32, j < 3 ? 2304 : 5120, 1) ||
            (a[j]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return false;
        if (j < 4 && !ggml_node_has_n_uses(&gpu.graph, at+j, 1)) return false;
        if (j < 3 && (a[j]->view_src || a[j]->view_offs || a[j]->extra || (a[j]->flags & (GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT)))) return false;
    }
    ggml_tensor * norm = a[0]->src[1];
    if (!prefix169_named(norm, "ffn_norm", layer) || !prefix169_same_device(device, norm) ||
        !prefix169_shape(norm, GGML_TYPE_F32, 5120, 1) || norm_source->data != norm->data ||
        !(norm_source == norm || (norm_source->view_src == norm && norm_source->view_offs == 0)) ||
        a[1]->src[1] != norm || a[2]->src[0] != a[0] || a[2]->src[1] != a[1] ||
        a[3]->src[1] != a[2] || a[4]->src[1] != a[3] || !prefix169_named(a[4]->src[0], "ffn_moe_out", layer) ||
        ggml_get_glu_op(a[2]) != GGML_GLU_OP_SWIGLU_CLAMP) return false;
    for (int j = 0; j < 3; ++j) {
        ggml_tensor * w = a[j == 2 ? 3 : j]->src[0];
        const char * part[] = {"gate","up","down"};
        if (!prefix169_weight(w, part[j], layer) || !prefix169_same_device(device, w) ||
            !prefix169_shape(w, j == 1 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q6_K, j == 2 ? 2304 : 5120, j == 2 ? 5120 : 2304)) return false;
    }
    // Identity, original DAG's global use counts, numerical operation order
    // within each branch and final ADD operands are retained in both views.
    p.cpu_split = ci;
    p.gpu_split = ci+1;
    p.device = device;
    p.early = ggml_graph_view(&gpu.graph, at, at+3);
    p.early.uid = ggml_graph_next_uid();
    p.tail_nodes.clear();
    p.tail_nodes.reserve(gpu.graph.n_nodes-3);
    p.tail_nodes.insert(p.tail_nodes.end(), gpu.graph.nodes, a);
    p.tail_nodes.insert(p.tail_nodes.end(), a+3, gpu.graph.nodes+gpu.graph.n_nodes);
    p.tail = ggml_graph_view(&gpu.graph, 0, gpu.graph.n_nodes);
    p.tail.nodes = p.tail_nodes.data();
    p.tail.n_nodes = int(p.tail_nodes.size());
    p.tail.uid = ggml_graph_next_uid();
    p.checked = {a[0], a[1], a[2], a[3], a[4], norm, a[0]->src[0], a[1]->src[0], a[3]->src[0]};
    for (size_t j = 0; j < p.checked.size(); ++j) {
        p.expected_data[j] = p.checked[j]->data;
        p.expected_buffer[j] = p.checked[j]->buffer;
    }
    return true;
}
static bool prefix169_bindings_match(const prefix169_pair & p) {
    for (size_t j = 0; j < p.checked.size(); ++j) {
        if (p.checked[j]->data != p.expected_data[j] || p.checked[j]->buffer != p.expected_buffer[j]) return false;
    }
    return true;
}
static prefix169_state * prefix169_prepare(ggml_backend_sched_t sched) {
    prefix169_state * s = sched->prefix169;
    if (!s || sched->callback_eval || sched->n_copies != 1 || sched->cur_copy != 0 ||
        sched->n_splits < 2 || !sched->splits[0].graph.uid) return nullptr;
    const uint64_t uid = sched->splits[0].graph.uid;
    if (s->last_graph_uid == uid) return s->enabled_graph ? s : nullptr;
    s->last_graph_uid = uid;
    s->enabled_graph = false;
    // Success requires one exact pair for every CPU-routed layer, in order.
    // The ordinary path is still valid after any failed match.
    unsigned count = 0;
    try {
        for (int ci = 0; ci + 1 < sched->n_splits && count < s->n_layers; ++ci) {
            if (prefix169_pair_prepare(sched, ci, int(count), s->pairs[count])) ++count;
        }
        if (count != s->n_layers || !prefix169_scratch(*s, s->pairs[0].device)) return nullptr;
        s->early_at.assign(sched->n_splits, nullptr);
        s->tail_at.assign(sched->n_splits, nullptr);
        for (auto & p : s->pairs) {
            if (p.device != s->device) return nullptr;
            s->early_at[p.cpu_split] = &p;
            s->tail_at[p.gpu_split] = &p;
        }
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
    ++s->plans;
    s->enabled_graph = true;
    return s;
}
struct prefix169_bind {
    ggml_tensor * tensor;
    void * data;
    ggml_backend_buffer_t buffer;
    prefix169_bind(ggml_tensor * t, const ggml_tensor * slot)
        : tensor(t), data(t->data), buffer(t->buffer) {
        t->data = slot->data;
        t->buffer = slot->buffer;
    }
    ~prefix169_bind() {
        tensor->data = data;
        tensor->buffer = buffer;
    }
    prefix169_bind(const prefix169_bind &) = delete;
    prefix169_bind & operator=(const prefix169_bind &) = delete;
};
struct prefix169_pending {
    prefix169_state * state = nullptr;
    ~prefix169_pending() {
        if (state) {
            ++state->fault_drains;
            ggml_backend_synchronize(state->device);
        }
    }
};
static ggml_status prefix169_dispatch(prefix169_state & s, prefix169_pending & pending, int split_id,
        ggml_backend_t backend, ggml_cgraph * original) {
    if (auto * p = s.early_at[split_id]) {
        GGML_ASSERT(pending.state == nullptr);
        if (!prefix169_bindings_match(*p)) return ggml_backend_graph_compute_async(backend, original);
        pending.state = &s;
        {
            prefix169_bind gate(p->early.nodes[0], s.scratch[0]);
            prefix169_bind up(p->early.nodes[1], s.scratch[1]);
            prefix169_bind glu(p->early.nodes[2], s.scratch[2]);
            ++s.early_submits;
            auto rc = ggml_backend_graph_compute_async(s.device, &p->early);
            if (rc != GGML_STATUS_SUCCESS) return rc;
        }
        return ggml_backend_graph_compute_async(backend, original);
    }
    if (auto * p = s.tail_at[split_id]) {
        if (!pending.state) return ggml_backend_graph_compute_async(backend, original);
        GGML_ASSERT(pending.state == &s);
        if (!prefix169_bindings_match(*p)) return GGML_STATUS_FAILED;
        prefix169_bind glu(p->early.nodes[2], s.scratch[2]);
        auto rc = ggml_backend_graph_compute_async(backend, &p->tail);
        if (rc == GGML_STATUS_SUCCESS) {
            ++s.tail_submits;
            pending.state = nullptr;
        }
        return rc;
    }
    return ggml_backend_graph_compute_async(backend, original);
}
extern "C" GGML_API int xeon_shared_prefix_stats_v1(ggml_backend_sched_t, uint64_t *, size_t);
extern "C" GGML_API int xeon_shared_prefix_stats_v1(ggml_backend_sched_t sched, uint64_t * out, size_t n) {
    if (!sched || !out || n < 7) return -1;
    const auto * s = sched->prefix169;
    std::fill(out, out+7, 0);
    if (s) {
        out[0] = 1;
        out[1] = s->plans;
        out[2] = s->early_submits;
        out[3] = s->tail_submits;
        out[4] = s->fault_drains;
        out[5] = s->scratch_buffer ? ggml_backend_buffer_get_size(s->scratch_buffer) : 0;
        out[6] = s->enabled_graph ? s->n_layers : 0;
    }
    return 0;
}

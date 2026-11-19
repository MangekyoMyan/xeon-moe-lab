#pragma once

// Review proposal for the d98a17c bank-FREE, original CPU graph only.
// This helper is not integrated, linked or run against the model.
// Never enable it in the performance arm: a non-null cb_eval changes scheduler
// synchronization even when no cell is armed. Allocate Capture outside decode.
// No file I/O, heap allocation, heap allocation in callback. This isolated diagnostic permits a 24-byte routing weight readback after normal callback backend sync.
// Caller supplies exact ORIGINAL model layer gate/down pointers and a unique
// case/call key. Attach labels, provider hashes, clamp and route37 generation
// to the resulting records OUTSIDE callback after normal call completion.

#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace bounded_moe_capture {
constexpr size_t max_cells = 8;
constexpr size_t n_embd = 5120;
constexpr size_t n_ff = 2304;
constexpr size_t n_used = 6;

enum : uint32_t {
    bad_shape = 1, not_host = 2, bad_ids = 4, duplicate = 8,
    incomplete = 16, invalid_setup = 32
};

struct Cell {
    uint64_t key = 0; // globally unique (case, T1 call), assigned by caller
    int layer = -1;
    const ggml_tensor * gate_weight = nullptr;
    const ggml_tensor * down_weight = nullptr;
    std::array<int32_t, n_used> expected_ids{};
    char weighted_name[GGML_MAX_NAME]{};
    char down_copy_name[GGML_MAX_NAME]{};
    char weights_name[GGML_MAX_NAME]{};
    const ggml_tensor * weights_producer=nullptr;
    bool weights_ready_after_gate=false;
    bool routing_weight_readback=false,down_via_cuda_copy=false;

    // Owned, bounded storage; exported only after caller synchronization.
    std::array<float, n_embd> x{};
    std::array<int32_t, n_used> ids{};
    std::array<float, n_embd * n_used> down{}; // slot-major, unweighted
    std::array<float, n_used> routing_weight{}; // effective norm+scale weights
    std::array<int64_t, 4> x_ne{}, x_nb{}, ids_ne{}, ids_nb{};
    uint32_t flags = 0;
    uint8_t seen = 0; // 1=input+IDs; 2=down; 4=effective router weights
    const ggml_tensor * down_node = nullptr; // valid only within armed call
};

struct Capture {
    std::array<Cell, max_cells> cells{};
    size_t count = 0;
    uint64_t current_key = 0;
    bool armed = false;
    ggml_backend_t gpu_backend=nullptr;

    // Run before model inference. Duplicate (key, layer) is prohibited.
    bool prepare(size_t n) {
        if (n > max_cells) return false;
        count = n;
        for (size_t i = 0; i < count; ++i) {
            auto & c = cells[i];
            if (c.layer < 0 || c.layer >= 37 || !c.gate_weight ||
                !c.down_weight || c.gate_weight == c.down_weight) return false;
            for (size_t j = 0; j < i; ++j)
                if (cells[j].key == c.key && cells[j].layer == c.layer) return false;
            for (size_t j = 0; j < n_used; ++j) {
                if (c.expected_ids[j] < 0 || c.expected_ids[j] >= 384) return false;
                for (size_t k = 0; k < j; ++k)
                    if (c.expected_ids[j] == c.expected_ids[k]) return false;
            }
            std::snprintf(c.weighted_name, sizeof(c.weighted_name),
                          "ffn_moe_weighted-%d", c.layer);
            std::snprintf(c.down_copy_name,sizeof(c.down_copy_name),"CUDA0#ffn_moe_down-%d#0",c.layer);
            std::snprintf(c.weights_name,sizeof(c.weights_name),"ffn_moe_weights_scaled-%d",c.layer);
        }
        return true;
    }

    // Only between completed llama calls, on the caller thread.
    void begin_call(uint64_t key) {
        current_key = key;
        armed = true;
        for (size_t i = 0; i < count; ++i) if (cells[i].key == key) {
            cells[i].flags = 0; cells[i].seen = 0; cells[i].down_node = nullptr;cells[i].weights_producer=nullptr;cells[i].weights_ready_after_gate=false;
        }
    }
    bool end_call() { // AFTER the existing normal completion/synchronization
        bool ok = true;
        for (size_t i = 0; i < count; ++i) if (cells[i].key == current_key) {
            if (cells[i].seen != 7) cells[i].flags |= incomplete;
            ok &= cells[i].flags == 0;
            cells[i].down_node = nullptr;
        }
        armed = false;
        return ok;
    }

    static bool host(const ggml_tensor * t) {
        if (!t || !t->data) return false;
        const auto * root = t;
        // A view may use its parent's buffer. Limit traversal on malformed data.
        for (int i = 0; !root->buffer && root->view_src && i < 8; ++i)
            root = root->view_src;
        return root->buffer && ggml_backend_buffer_is_host(root->buffer);
    }
    static bool shape(const ggml_tensor * t, ggml_type type,
                      int64_t a, int64_t b, int64_t c, int64_t d) {
        return t && t->type == type && t->ne[0] == a && t->ne[1] == b &&
            t->ne[2] == c && t->ne[3] == d && ggml_is_contiguous(t);
    }
    static int match(const Cell & c, const ggml_tensor * t) {
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] == c.gate_weight) return 1;
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] == c.down_weight) return 2;
        if (t->op == GGML_OP_MUL && c.down_node && t->src[0] &&
            (t->src[0] == c.down_node ||
             (t->src[0]->op == GGML_OP_NONE && std::strcmp(t->src[0]->name,c.down_copy_name)==0 &&
              shape(t->src[0],GGML_TYPE_F32,n_embd,n_used,1,1))) &&
            std::strcmp(t->name, c.weighted_name) == 0) return 4;
        return 0;
    }
    static bool callback(ggml_tensor * t, bool ask, void * opaque) {
        auto & s = *static_cast<Capture *>(opaque);
        if (!s.armed) return ask ? false : true;
        bool wanted = false;
        for (size_t i = 0; i < s.count; ++i) {
            auto & c = s.cells[i];
            if (c.key != s.current_key) continue;
            // Remember the actual scaled weights producer without requesting any
            // CUDA graph cut. This must be visited before the CPU gate executes.
            if(ask && std::strcmp(t->name,c.weights_name)==0){
                if(c.weights_producer || c.seen || t->op!=GGML_OP_SCALE ||
                   !shape(t,GGML_TYPE_F32,1,n_used,1,1))c.flags|=invalid_setup;
                else c.weights_producer=t;
                continue;
            }
            const int what = match(c, t);
            if (!what) continue;
            // Validate the consumer uses exactly the already-copied producer.
            // Return false: preserve the original CUDA weighted/merge graph.
            if(what==4){
                if(!ask || c.seen!=3 || !c.weights_ready_after_gate || t->src[1]!=c.weights_producer){c.flags|=invalid_setup;continue;}
                c.down_via_cuda_copy=t->src[0]!=c.down_node;c.seen|=4;continue;
            }
            wanted = true;
            if (ask) continue;
            if (c.seen & what) { c.flags |= duplicate; continue; }
            if (what == 1) {
                const auto * x = t->src[1]; const auto * ids = t->src[2];
                if (!shape(t, GGML_TYPE_F32, n_ff, n_used, 1, 1) ||
                    !shape(x, GGML_TYPE_F32, n_embd, 1, 1, 1) ||
                    !shape(ids, GGML_TYPE_I32, n_used, 1, 1, 1)) {
                    c.flags |= bad_shape; continue;
                }
                if (!host(x) || !host(ids)) { c.flags |= not_host; continue; }
                std::memcpy(c.x.data(), x->data, sizeof(c.x));
                std::memcpy(c.ids.data(), ids->data, sizeof(c.ids));
                for (size_t d = 0; d < 4; ++d) {
                    c.x_ne[d] = x->ne[d]; c.x_nb[d] = int64_t(x->nb[d]);
                    c.ids_ne[d] = ids->ne[d]; c.ids_nb[d] = int64_t(ids->nb[d]);
                }
                if (c.ids != c.expected_ids) c.flags |= bad_ids;
                // In this fixed model scaled weights are expanded before CPU FFN.
                // Its producer was encountered in a prior GPU split (or an earlier
                // CPU node). The CPU gate has now completed. Explicitly drain the
                // original GPU backend before the diagnostic 24-byte read, avoiding
                // assumptions about cudaStreamPerThread vs the scheduler stream.
                const auto *w=c.weights_producer;
                if(!w || !w->data || !s.gpu_backend){c.flags|=invalid_setup;continue;}
                if(host(w))std::memcpy(c.routing_weight.data(),w->data,sizeof(c.routing_weight));
                else{ggml_backend_synchronize(s.gpu_backend);c.routing_weight_readback=true;
                    ggml_backend_tensor_get(w,c.routing_weight.data(),0,sizeof(c.routing_weight));}
                c.weights_ready_after_gate=true;
            } else if (what == 2) {
                if (!shape(t, GGML_TYPE_F32, n_embd, n_used, 1, 1)) {
                    c.flags |= bad_shape; continue;
                }
                if (!host(t)) { c.flags |= not_host; continue; }
                std::memcpy(c.down.data(), t->data, sizeof(c.down));
                c.down_node = t;
            }
            c.seen |= uint8_t(what);
        }
        // Never use false observation return as a global cancellation mechanism.
        // Save failure flags, then fail the caller AFTER normal completion.
        return ask ? wanted : true;
    }
};
} // namespace bounded_moe_capture
